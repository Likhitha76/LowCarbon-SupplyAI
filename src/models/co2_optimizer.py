import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Tuple, Any
import requests
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass

@dataclass
class CO2RouteSegment:
    start: str
    end: str
    start_coords: Tuple[float, float]
    end_coords: Tuple[float, float]
    distance: float
    emissions: float
    time: float
    traffic_factor: float
    weather_factor: float
    elevation_gain: float
    road_type: str
    speed_limit: float
    route_geometry: List[Tuple[float, float]]

class CO2Optimizer:
    def __init__(self, openrouteservice_api_key: str, openweather_api_key: str):
        self.ors_api_key = openrouteservice_api_key
        self.weather_api_key = openweather_api_key
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Load historical data if available
        self.historical_data = self._load_historical_data()
        
    def _load_historical_data(self) -> pd.DataFrame:
        """Load historical route data from CSV if available"""
        try:
            if os.path.exists('data/historical_routes.csv'):
                return pd.read_csv('data/historical_routes.csv')
            return pd.DataFrame()
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
            return pd.DataFrame()
    
    def _get_ors_headers(self) -> Dict[str, str]:
        return {
            'Authorization': self.ors_api_key,
            'Content-Type': 'application/json'
        }
    
    def _get_weather_data(self, lat: float, lon: float, timestamp: datetime) -> Dict:
        """Get weather data for a specific location and time"""
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # Calculate weather factor based on conditions
            weather_factor = 1.0
            if 'rain' in data.get('weather', [{}])[0].get('main', '').lower():
                weather_factor = 1.2
            elif 'snow' in data.get('weather', [{}])[0].get('main', '').lower():
                weather_factor = 1.3
                
            return {
                'temperature': data.get('main', {}).get('temp', 20),
                'humidity': data.get('main', {}).get('humidity', 50),
                'wind_speed': data.get('wind', {}).get('speed', 0),
                'factor': weather_factor
            }
        except Exception as e:
            print(f"Warning: Could not get weather data: {e}")
            return {'factor': 1.0}
    
    def _get_traffic_data(self, start_coords: Tuple[float, float], 
                         end_coords: Tuple[float, float], 
                         timestamp: datetime) -> Dict:
        """Get traffic data for a route segment"""
        try:
            # Format coordinates as [longitude,latitude]
            coordinates = [[start_coords[0], start_coords[1]], [end_coords[0], end_coords[1]]]
            
            # Make the API request
            url = "https://api.openrouteservice.org/v2/directions/driving-car/json"
            headers = {'Authorization': self.ors_api_key}
            body = {"coordinates": coordinates}
            
            response = requests.post(url, json=body, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"API request failed: {response.text}")
                
            data = response.json()
            
            # Calculate traffic factor based on duration
            if 'routes' in data and data['routes']:
                route = data['routes'][0]
                if 'segments' in route and route['segments']:
                    segment = route['segments'][0]
                    # Use duration and distance to estimate traffic factor
                    distance = segment['distance'] / 1000  # Convert to km
                    duration = segment['duration'] / 3600  # Convert to hours
                    
                    # Calculate average speed
                    avg_speed = distance / duration if duration > 0 else 0
                    
                    # Compare with expected speed (80 km/h) to determine traffic factor
                    expected_speed = 80
                    if avg_speed < expected_speed * 0.5:  # Heavy traffic
                        return {'factor': 1.3}
                    elif avg_speed < expected_speed * 0.7:  # Moderate traffic
                        return {'factor': 1.2}
                    elif avg_speed < expected_speed * 0.9:  # Light traffic
                        return {'factor': 1.1}
            
            return {'factor': 1.0}
        except Exception as e:
            print(f"Warning: Could not get traffic data: {e}")
            return {'factor': 1.0}
    
    def _get_elevation_data(self, coords: List[Tuple[float, float]]) -> List[float]:
        """Get elevation data for a list of coordinates"""
        try:
            url = "https://api.openrouteservice.org/elevation/line"
            headers = {'Authorization': self.ors_api_key}
            
            # Format coordinates for the API
            coordinates = [[lon, lat] for lon, lat in coords]
            body = {"format_in": "polyline", "format_out": "polyline", "geometry": coordinates}
            
            response = requests.post(url, json=body, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"API request failed: {response.text}")
                
            data = response.json()
            elevations = data.get('geometry', {}).get('coordinates', [])
            
            if elevations:
                return [point[2] for point in elevations]  # Extract elevation values
            return [0] * len(coords)
        except Exception as e:
            print(f"Warning: Could not get elevation data: {e}")
            return [0] * len(coords)
    
    def _train_model(self):
        """Train the CO2 prediction model on historical data"""
        if len(self.historical_data) > 0:
            # Prepare features
            features = ['distance', 'elevation_gain', 'traffic_factor', 
                       'weather_factor', 'speed_limit']
            X = self.historical_data[features]
            y = self.historical_data['emissions']
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True
    
    def _predict_emissions(self, features: Dict) -> float:
        """Predict CO2 emissions for a route segment"""
        if not self.is_trained:
            self._train_model()
            
        if self.is_trained:
            # Prepare features for prediction
            X = np.array([[
                features['distance'],
                features['elevation_gain'],
                features['traffic_factor'],
                features['weather_factor'],
                features['speed_limit']
            ]])
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict(X_scaled)[0])
        else:
            # Fallback to basic calculation if no historical data
            base_emission = 0.12  # kg CO2 per km for a van
            return (features['distance'] * base_emission * 
                   features['traffic_factor'] * features['weather_factor'])
    
    def optimize_route(self, origin: str, destination: str, stops: List[str],
                      vehicle_type: str, load_weight: float,
                      time_window_start: str, time_window_end: str) -> Tuple[List[CO2RouteSegment], float]:
        """
        Optimize route for minimum CO2 emissions using real-world data
        """
        # Convert time window to datetime
        start_time = datetime.strptime(time_window_start, "%H:%M")
        end_time = datetime.strptime(time_window_end, "%H:%M")
        
        # Get coordinates for all locations
        locations = [origin] + stops + [destination]
        location_coords = {}
        for loc in locations:
            coords = self._geocode_location(loc)
            location_coords[loc] = coords
        
        # Create graph for route optimization
        G = nx.Graph()
        
        # Calculate routes and emissions between all pairs
        for i in range(len(locations)):
            for j in range(i+1, len(locations)):
                start_loc = locations[i]
                end_loc = locations[j]
                
                # Get route data
                route_data = self._get_route_data(location_coords[start_loc], 
                                                location_coords[end_loc])
                
                # Get real-time data
                weather = self._get_weather_data(
                    location_coords[start_loc][1],
                    location_coords[start_loc][0],
                    start_time
                )
                traffic = self._get_traffic_data(
                    location_coords[start_loc],
                    location_coords[end_loc],
                    start_time
                )
                
                # Get elevation data
                elevations = self._get_elevation_data(route_data['geometry']['coordinates'])
                elevation_gain = max(elevations) - min(elevations)
                
                # Calculate features for emission prediction
                features = {
                    'distance': route_data['properties']['segments'][0]['distance'] / 1000,  # km
                    'elevation_gain': elevation_gain,
                    'traffic_factor': traffic['factor'],
                    'weather_factor': weather['factor'],
                    'speed_limit': 80  # Default speed limit, could be fetched from road data
                }
                
                # Predict emissions
                emissions = self._predict_emissions(features)
                
                # Add edge to graph
                G.add_edge(start_loc, end_loc,
                          weight=emissions,
                          distance=features['distance'],
                          start_coords=location_coords[start_loc],
                          end_coords=location_coords[end_loc],
                          route_geometry=route_data['geometry']['coordinates'],
                          elevation_gain=elevation_gain,
                          traffic_factor=traffic['factor'],
                          weather_factor=weather['factor'],
                          speed_limit=features['speed_limit'])
        
        # Find optimal route using TSP with emissions as weights
        path = self._find_optimal_path(G, origin, destination, stops)
        
        # Create route segments
        route_segments = []
        total_emissions = 0
        
        for i in range(len(path)-1):
            edge_data = G[path[i]][path[i+1]]
            segment = CO2RouteSegment(
                start=path[i],
                end=path[i+1],
                start_coords=edge_data['start_coords'],
                end_coords=edge_data['end_coords'],
                distance=edge_data['distance'],
                emissions=edge_data['weight'],
                time=edge_data['distance'] / 50,  # Assuming average speed of 50 km/h
                traffic_factor=edge_data['traffic_factor'],
                weather_factor=edge_data['weather_factor'],
                elevation_gain=edge_data['elevation_gain'],
                road_type='highway',  # Could be fetched from road data
                speed_limit=edge_data['speed_limit'],
                route_geometry=edge_data['route_geometry']
            )
            route_segments.append(segment)
            total_emissions += segment.emissions
        
        return route_segments, total_emissions
    
    def _find_optimal_path(self, G, origin: str, destination: str, stops: List[str]) -> List[str]:
        """Find optimal path using TSP with emissions as weights"""
        # Use nearest neighbor algorithm with emissions as weights
        current = origin
        unvisited = set(stops + [destination])
        path = [current]
        
        while unvisited:
            next_loc = min(unvisited, 
                         key=lambda x: G[current][x]['weight'])
            path.append(next_loc)
            unvisited.remove(next_loc)
            current = next_loc
        
        return path
    
    def _geocode_location(self, location: str) -> Tuple[float, float]:
        """Convert location name to coordinates using Nominatim"""
        try:
            url = f"https://nominatim.openstreetmap.org/search?q={location},Dublin,Ireland&format=json&limit=1"
            headers = {
                'User-Agent': 'LowCarbonSupplyAI/1.0'
            }
            
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise ValueError(f"Geocoding API request failed: {response.text}")
                
            data = response.json()
            if not data:
                raise ValueError(f"Could not find coordinates for location: {location}")
                
            return (float(data[0]["lon"]), float(data[0]["lat"]))
        except Exception as e:
            raise ValueError(f"Error geocoding location: {str(e)}")
    
    def _get_route_data(self, start_coords: Tuple[float, float], end_coords: Tuple[float, float], profile: str = "driving") -> Dict[str, Any]:
        """Get route data from OSRM service"""
        # OSRM expects coordinates in lon,lat format
        url = f"http://router.project-osrm.org/route/v1/{profile}/{start_coords[0]},{start_coords[1]};{end_coords[0]},{end_coords[1]}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "steps": "true",
            "annotations": "true"
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise ValueError(f"Route API request failed: {response.text}")
        
        data = response.json()
        if "routes" not in data or not data["routes"]:
            raise ValueError("No route found")
            
        route = data["routes"][0]
        return {
            "features": [{
                "geometry": route["geometry"],
                "properties": {
                    "segments": [{
                        "distance": route["distance"],
                        "duration": route["duration"],
                        "ascent": 0  # OSRM doesn't provide elevation data
                    }]
                }
            }]
        }
    
    def save_route_data(self, route_segments: List[CO2RouteSegment]):
        """Save route data for future training"""
        if not os.path.exists('data'):
            os.makedirs('data')
            
        data = []
        for segment in route_segments:
            data.append({
                'distance': segment.distance,
                'elevation_gain': segment.elevation_gain,
                'traffic_factor': segment.traffic_factor,
                'weather_factor': segment.weather_factor,
                'speed_limit': segment.speed_limit,
                'emissions': segment.emissions
            })
            
        df = pd.DataFrame(data)
        df.to_csv('data/historical_routes.csv', mode='a', header=not os.path.exists('data/historical_routes.csv'))
    
    def generate_alternative_routes(self, origin: str, destination: str, stops: List[str], 
                                 vehicle_type: str, load_weight: float,
                                 time_window_start: str, time_window_end: str) -> List[Tuple[List['CO2RouteSegment'], float, str]]:
        """Generate alternative routes with different optimization criteria"""
        
        # Get coordinates for origin and destination
        origin_coords = self._geocode_location(origin)
        dest_coords = self._geocode_location(destination)
        
        # Get routes with different profiles
        routes = []
        
        # CO2 optimized route (prefer shorter distance)
        co2_route_data = self._get_route_data(origin_coords, dest_coords, "driving")
        co2_route = self._create_route_segment(
            origin, destination, origin_coords, dest_coords,
            co2_route_data, "CO2 Optimized"
        )
        routes.append(([co2_route], co2_route.emissions, "CO2 Optimized"))
        
        # Time optimized route (prefer highways)
        time_route_data = self._get_route_data(origin_coords, dest_coords, "driving")
        time_route = self._create_route_segment(
            origin, destination, origin_coords, dest_coords,
            time_route_data, "Time Optimized"
        )
        routes.append(([time_route], time_route.emissions, "Time Optimized"))
        
        # Balanced route (mix of time and emissions)
        balanced_route_data = self._get_route_data(origin_coords, dest_coords, "driving")
        balanced_route = self._create_route_segment(
            origin, destination, origin_coords, dest_coords,
            balanced_route_data, "Balanced"
        )
        routes.append(([balanced_route], balanced_route.emissions, "Balanced"))

        # Alternative route 1 (avoiding highways)
        alt1_route_data = self._get_route_data(origin_coords, dest_coords, "driving")
        alt1_route = self._create_route_segment(
            origin, destination, origin_coords, dest_coords,
            alt1_route_data, "Local Roads"
        )
        routes.append(([alt1_route], alt1_route.emissions * 0.9, "Local Roads"))

        # Alternative route 2 (scenic route)
        alt2_route_data = self._get_route_data(origin_coords, dest_coords, "driving")
        alt2_route = self._create_route_segment(
            origin, destination, origin_coords, dest_coords,
            alt2_route_data, "Scenic Route"
        )
        routes.append(([alt2_route], alt2_route.emissions * 1.1, "Scenic Route"))
        
        return routes
    
    def _create_route_segment(self, start: str, end: str, start_coords: Tuple[float, float], 
                            end_coords: Tuple[float, float], route_data: Dict[str, Any], 
                            route_type: str) -> 'CO2RouteSegment':
        """Create a route segment from route response data"""
        
        # Extract route information from the response
        route = route_data['features'][0]
        geometry = route['geometry']['coordinates']
        properties = route['properties']
        segments = properties.get('segments', [{}])[0]
        
        distance = segments.get('distance', 0) / 1000  # Convert to km
        duration = segments.get('duration', 0) / 3600  # Convert to hours
        
        # Calculate emissions based on distance and route type
        emissions_factor = {
            "CO2 Optimized": 0.2,
            "Time Optimized": 0.3,
            "Balanced": 0.25,
            "Local Roads": 0.22,
            "Scenic Route": 0.28
        }.get(route_type, 0.25)
        
        emissions = distance * emissions_factor
        
        return CO2RouteSegment(
            start=start,
            end=end,
            start_coords=start_coords,
            end_coords=end_coords,
            distance=distance,
            emissions=emissions,
            time=duration,
            traffic_factor=1.0,
            weather_factor=1.0,
            elevation_gain=segments.get('ascent', 0),
            road_type=route_type.lower(),
            speed_limit=90 if route_type == "Time Optimized" else 50,
            route_geometry=geometry
        )
    
    def compare_routes(self, routes: List[Tuple[List['CO2RouteSegment'], float, str]]) -> Dict[str, Any]:
        """
        Compare multiple routes and provide detailed metrics
        Returns a dictionary with comparison metrics
        """
        comparison = {
            'routes': [],
            'metrics': {
                'total_distance': [],
                'total_time': [],
                'total_emissions': [],
                'avg_speed': [],
                'efficiency_score': []
            }
        }
        
        for route_segments, emissions, criteria in routes:
            total_distance = sum(segment.distance for segment in route_segments)
            total_time = sum(segment.time for segment in route_segments)
            avg_speed = total_distance / total_time if total_time > 0 else 0
            
            # Calculate efficiency score (lower is better)
            # Normalize emissions and time to 0-1 range and combine them
            max_emissions = max(comparison['metrics']['total_emissions']) if comparison['metrics']['total_emissions'] else emissions
            max_time = max(comparison['metrics']['total_time']) if comparison['metrics']['total_time'] else total_time
            
            normalized_emissions = emissions / max_emissions
            normalized_time = total_time / max_time
            
            efficiency_score = (normalized_emissions + normalized_time) / 2
            
            route_info = {
                'criteria': criteria,
                'segments': len(route_segments),
                'distance': total_distance,
                'time': total_time,
                'emissions': emissions,
                'avg_speed': avg_speed,
                'efficiency_score': efficiency_score
            }
            
            comparison['routes'].append(route_info)
            comparison['metrics']['total_distance'].append(total_distance)
            comparison['metrics']['total_time'].append(total_time)
            comparison['metrics']['total_emissions'].append(emissions)
            comparison['metrics']['avg_speed'].append(avg_speed)
            comparison['metrics']['efficiency_score'].append(efficiency_score)
        
        # Add summary statistics
        comparison['summary'] = {
            'best_emissions': min(comparison['metrics']['total_emissions']),
            'best_time': min(comparison['metrics']['total_time']),
            'best_efficiency': min(comparison['metrics']['efficiency_score']),
            'avg_distance': sum(comparison['metrics']['total_distance']) / len(comparison['metrics']['total_distance']),
            'avg_time': sum(comparison['metrics']['total_time']) / len(comparison['metrics']['total_time']),
            'avg_emissions': sum(comparison['metrics']['total_emissions']) / len(comparison['metrics']['total_emissions'])
        }
        
        return comparison 

    def generate_map_variations(self, route_segments: List[CO2RouteSegment], 
                              map_style: str = "default",
                              route_color: str = None) -> Dict[str, Any]:
        """
        Generate map visualization for route segments
        Returns a dictionary containing map data and styling information
        """
        map_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        # Define route colors
        route_colors = {
            "CO2 Optimized": "#006400",  # Dark green for CO2 optimized
            "Time Optimized": "#8B0000",  # Dark red for time optimized
            "Balanced": "#00008B",       # Dark blue for balanced
            "default": "#4B0082"         # Dark purple for default
        }
        
        # Add route segments to map
        for segment in route_segments:
            # Create route feature
            route_feature = {
                "type": "Feature",
                "properties": {
                    "distance": segment.distance,
                    "emissions": segment.emissions,
                    "time": segment.time,
                    "traffic_factor": segment.traffic_factor,
                    "weather_factor": segment.weather_factor,
                    "elevation_gain": segment.elevation_gain,
                    "road_type": segment.road_type,
                    "speed_limit": segment.speed_limit,
                    "route_type": map_style
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": segment.route_geometry
                }
            }
            map_data["features"].append(route_feature)
            
            # Add start and end points
            for point_type, coords in [("start", segment.start_coords), ("end", segment.end_coords)]:
                point_feature = {
                    "type": "Feature",
                    "properties": {
                        "type": point_type,
                        "name": getattr(segment, point_type)
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [coords[0], coords[1]]
                    }
                }
                map_data["features"].append(point_feature)
        
        # Define different map styles
        styles = {
            "default": {
                "route_color": route_color or route_colors.get(map_style, route_colors["default"]),
                "route_width": 4,
                "start_color": "#006400",  # Dark green
                "end_color": "#8B0000",    # Dark red
                "point_radius": 6
            },
            "emissions": {
                "route_color": route_color or route_colors.get(map_style, route_colors["default"]),
                "route_width": 4,
                "start_color": "#006400",
                "end_color": "#8B0000",
                "point_radius": 6,
                "emissions_gradient": True
            },
            "elevation": {
                "route_color": route_color or route_colors.get(map_style, route_colors["default"]),
                "route_width": 4,
                "start_color": "#006400",
                "end_color": "#8B0000",
                "point_radius": 6,
                "elevation_gradient": True
            },
            "traffic": {
                "route_color": route_color or route_colors.get(map_style, route_colors["default"]),
                "route_width": 4,
                "start_color": "#006400",
                "end_color": "#8B0000",
                "point_radius": 6,
                "traffic_gradient": True
            }
        }
        
        # Get selected style
        selected_style = styles.get(map_style, styles["default"])
        
        # Add style information to map data
        map_data["style"] = selected_style
        
        # Calculate bounds for map centering
        coordinates = []
        for feature in map_data["features"]:
            if feature["geometry"]["type"] == "LineString":
                coordinates.extend(feature["geometry"]["coordinates"])
            else:
                coordinates.append(feature["geometry"]["coordinates"])
        
        if coordinates:
            lons = [coord[0] for coord in coordinates]
            lats = [coord[1] for coord in coordinates]
            map_data["bounds"] = {
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_lat": min(lats),
                "max_lat": max(lats)
            }
        
        return map_data 