import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
import openrouteservice
from dataclasses import dataclass
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
from .scenario_gan import ScenarioGAN

@dataclass
class RouteSegment:
    start: str
    end: str
    start_coords: Tuple[float, float]
    end_coords: Tuple[float, float]
    distance: float
    emissions: float
    time: float
    traffic_factor: float
    weather_factor: float
    route_geometry: List[Tuple[float, float]]  # List of coordinates for the actual route

class RouteOptimizer:
    def __init__(self, api_key: str):
        self.client = openrouteservice.Client(key=api_key)
        self.geocoder = Nominatim(user_agent="low_carbon_supply_chain")
        self.scenario_gan = ScenarioGAN()
        
    def geocode_location(self, location: str, max_retries: int = 3) -> Tuple[float, float]:
        """
        Convert location name to coordinates (longitude, latitude)
        """
        for attempt in range(max_retries):
            try:
                location_data = self.geocoder.geocode(location)
                if location_data is None:
                    raise ValueError(f"Could not find coordinates for location: {location}")
                # Return as (longitude, latitude) for OpenRouteService
                return (location_data.longitude, location_data.latitude)
            except GeocoderTimedOut:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1)
        
    def calculate_emissions(self, distance: float, vehicle_type: str, load_weight: float,
                          traffic_factor: float = 1.0, weather_factor: float = 1.0) -> float:
        """
        Calculate CO2 emissions for a given route segment
        """
        base_emission_factor = {
            'truck': 0.15,  # kg CO2 per km
            'van': 0.12,
            'car': 0.08
        }
        
        # Adjust emissions based on load weight (assuming linear relationship)
        load_factor = 1 + (load_weight / 1000)  # Assuming 1000kg as reference
        
        # Calculate total emissions
        emissions = (distance * base_emission_factor[vehicle_type] * 
                    load_factor * traffic_factor * weather_factor)
        
        return emissions
    
    def get_route_data(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> Dict:
        """
        Get route data from OpenRouteService API using coordinates
        """
        try:
            route = self.client.directions(
                coordinates=[list(start_coords), list(end_coords)],
                profile='driving-hgv',
                format='geojson'
            )
            return route
        except Exception as e:
            raise ValueError(f"Error getting route data: {str(e)}")
    
    def optimize_route(self, origin: str, destination: str, stops: List[str],
                      vehicle_type: str, load_weight: float,
                      time_window_start: str, time_window_end: str,
                      weather_conditions: Dict = None,
                      traffic_conditions: Dict = None) -> Tuple[List[RouteSegment], float]:
        """
        Optimize route considering multiple stops and constraints
        """
        print("Converting locations to coordinates...")
        
        # Convert all locations to coordinates
        locations = [origin] + stops + [destination]
        location_coords = {}
        for loc in locations:
            coords = self.geocode_location(loc)
            location_coords[loc] = coords
            print(f"Found coordinates for {loc}: {coords}")
        
        # Create graph for route optimization
        G = nx.Graph()
        
        print("\nCalculating routes between locations...")
        # Get route data between all pairs of locations
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                start_loc = locations[i]
                end_loc = locations[j]
                print(f"Calculating route: {start_loc} -> {end_loc}")
                
                route_data = self.get_route_data(location_coords[start_loc], location_coords[end_loc])
                distance = route_data['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert to km
                
                # Extract route geometry
                route_geometry = route_data['features'][0]['geometry']['coordinates']
                
                # Calculate emissions for this segment
                emissions = self.calculate_emissions(
                    distance, vehicle_type, load_weight,
                    traffic_conditions.get('factor', 1.0) if traffic_conditions else 1.0,
                    weather_conditions.get('factor', 1.0) if weather_conditions else 1.0
                )
                
                # Add edge to graph
                G.add_edge(start_loc, end_loc,
                          weight=emissions,
                          distance=distance,
                          start_coords=location_coords[start_loc],
                          end_coords=location_coords[end_loc],
                          route_geometry=route_geometry)
        
        # Find optimal route using TSP (Traveling Salesman Problem)
        # First, create a list of all locations that need to be visited
        locations_to_visit = [origin] + stops + [destination]
        
        # Find the optimal order of stops using nearest neighbor algorithm
        current = origin
        unvisited = set(locations_to_visit[1:])  # Exclude origin
        path = [current]
        
        while unvisited:
            # Find the nearest unvisited location
            next_loc = min(unvisited, 
                         key=lambda x: G[current][x]['weight'])
            path.append(next_loc)
            unvisited.remove(next_loc)
            current = next_loc
        
        # Calculate total emissions and create route segments
        total_emissions = 0
        route_segments = []
        
        for i in range(len(path)-1):
            edge_data = G[path[i]][path[i+1]]
            segment = RouteSegment(
                start=path[i],
                end=path[i+1],
                start_coords=edge_data['start_coords'],
                end_coords=edge_data['end_coords'],
                distance=edge_data['distance'],
                emissions=edge_data['weight'],
                time=edge_data['distance'] / 50,  # Assuming average speed of 50 km/h
                traffic_factor=traffic_conditions.get('factor', 1.0) if traffic_conditions else 1.0,
                weather_factor=weather_conditions.get('factor', 1.0) if weather_conditions else 1.0,
                route_geometry=edge_data['route_geometry']
            )
            route_segments.append(segment)
            total_emissions += segment.emissions
        
        return route_segments, total_emissions
    
    def generate_future_scenarios(self, current_route: List[RouteSegment], num_scenarios: int = 5):
        """
        Generate future scenarios using GAN
        """
        # Prepare current route data for GAN
        route_data = np.array([
            [seg.distance, seg.emissions, seg.traffic_factor, seg.weather_factor]
            for seg in current_route
        ])
        
        # Train GAN on current route data
        self.scenario_gan.train(route_data)
        
        # Generate future scenarios
        scenarios = self.scenario_gan.generate_scenarios(num_scenarios)
        
        return scenarios 