import networkx as nx
import numpy as np
from typing import List, Dict, Tuple
import openrouteservice
from dataclasses import dataclass
from .scenario_gan import ScenarioGAN

@dataclass
class RouteSegment:
    start: str
    end: str
    distance: float
    emissions: float
    time: float
    traffic_factor: float
    weather_factor: float

class RouteOptimizer:
    def __init__(self, api_key: str):
        self.client = openrouteservice.Client(key=api_key)
        self.scenario_gan = ScenarioGAN()
        
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
    
    def get_route_data(self, origin: str, destination: str) -> Dict:
        """
        Get route data from OpenRouteService API
        """
        try:
            route = self.client.directions(
                coordinates=[[origin], [destination]],
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
        # Create graph for route optimization
        G = nx.Graph()
        
        # Add all locations to graph
        locations = [origin] + stops + [destination]
        
        # Get route data between all pairs of locations
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                route_data = self.get_route_data(locations[i], locations[j])
                distance = route_data['features'][0]['properties']['segments'][0]['distance'] / 1000  # Convert to km
                
                # Calculate emissions for this segment
                emissions = self.calculate_emissions(
                    distance, vehicle_type, load_weight,
                    traffic_conditions.get('factor', 1.0) if traffic_conditions else 1.0,
                    weather_conditions.get('factor', 1.0) if weather_conditions else 1.0
                )
                
                # Add edge to graph
                G.add_edge(locations[i], locations[j],
                          weight=emissions,
                          distance=distance)
        
        # Find optimal route using Dijkstra's algorithm
        try:
            path = nx.shortest_path(G, source=origin, target=destination, weight='weight')
        except nx.NetworkXNoPath:
            raise ValueError("No valid route found between origin and destination")
        
        # Calculate total emissions
        total_emissions = 0
        route_segments = []
        
        for i in range(len(path)-1):
            edge_data = G[path[i]][path[i+1]]
            segment = RouteSegment(
                start=path[i],
                end=path[i+1],
                distance=edge_data['distance'],
                emissions=edge_data['weight'],
                time=edge_data['distance'] / 50,  # Assuming average speed of 50 km/h
                traffic_factor=traffic_conditions.get('factor', 1.0) if traffic_conditions else 1.0,
                weather_factor=weather_conditions.get('factor', 1.0) if weather_conditions else 1.0
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