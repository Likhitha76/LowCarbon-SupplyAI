from src.models.co2_optimizer import CO2Optimizer, CO2RouteSegment
import json
import sys

def main():
    # Check if origin and destination are provided
    if len(sys.argv) < 3:
        print("Usage: python find_route.py <origin> <destination>")
        print("Example: python find_route.py \"Drumcondra\" \"Phibsborough\"")
        sys.exit(1)
    
    origin = sys.argv[1]
    destination = sys.argv[2]
    
    # Initialize the optimizer with the OpenRouteService API key
    optimizer = CO2Optimizer(
        openrouteservice_api_key='5b3ce3597851110001cf6248e4f8b61a',  # Updated API key
        openweather_api_key=''  # Empty string for now
    )
    
    try:
        print(f"\nFinding routes from {origin} to {destination}")
        
        # Generate alternative routes using the OpenRouteService API
        routes = optimizer.generate_alternative_routes(
            origin=origin,
            destination=destination,
            stops=[],  # No intermediate stops
            vehicle_type="van",
            load_weight=500,  # kg
            time_window_start="09:00",
            time_window_end="17:00"
        )
        
        # Compare routes
        comparison = optimizer.compare_routes(routes)
        
        # Print route comparison
        print('\nRoute Comparison:')
        for route in comparison['routes']:
            print(f"\n{route['criteria']} Route:")
            print(f"Distance: {route['distance']:.2f} km")
            print(f"Time: {route['time']:.2f} hours")
            print(f"Emissions: {route['emissions']:.2f} kg CO2")
            print(f"Efficiency Score: {route['efficiency_score']:.2f}")
        
        # Print summary
        print(f"\nSummary:")
        print(f"Best Emissions: {comparison['summary']['best_emissions']:.2f} kg CO2")
        print(f"Best Time: {comparison['summary']['best_time']:.2f} hours")
        print(f"Best Efficiency: {comparison['summary']['best_efficiency']:.2f}")
        
        # Generate a single map with all routes
        combined_map_data = {
            "type": "FeatureCollection",
            "features": [],
            "styles": {
                "CO2 Optimized": {
                    "route_color": "#006400",  # Dark green
                    "route_width": 5,
                    "start_color": "#006400",
                    "end_color": "#8B0000",
                    "point_radius": 8
                },
                "Time Optimized": {
                    "route_color": "#8B0000",  # Dark red
                    "route_width": 5,
                    "start_color": "#006400",
                    "end_color": "#8B0000",
                    "point_radius": 8
                },
                "Balanced": {
                    "route_color": "#00008B",  # Dark blue
                    "route_width": 5,
                    "start_color": "#006400",
                    "end_color": "#8B0000",
                    "point_radius": 8
                }
            }
        }
        
        # Add each route to the combined map
        for route_segments, emissions, criteria in routes:
            for segment in route_segments:
                # Add route line
                feature = {
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
                        "route_type": criteria
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": segment.route_geometry
                    }
                }
                combined_map_data["features"].append(feature)
                
                # Add start point
                start_point = {
                    "type": "Feature",
                    "properties": {
                        "type": "start",
                        "name": segment.start
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": list(segment.start_coords)
                    }
                }
                combined_map_data["features"].append(start_point)
                
                # Add end point
                end_point = {
                    "type": "Feature",
                    "properties": {
                        "type": "end",
                        "name": segment.end
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": list(segment.end_coords)
                    }
                }
                combined_map_data["features"].append(end_point)
        
        # Calculate bounds
        coordinates = []
        for feature in combined_map_data["features"]:
            if feature["geometry"]["type"] == "LineString":
                coordinates.extend(feature["geometry"]["coordinates"])
            else:
                coordinates.append(feature["geometry"]["coordinates"])
        
        if coordinates:
            lons = [coord[0] for coord in coordinates]
            lats = [coord[1] for coord in coordinates]
            combined_map_data["bounds"] = {
                "min_lon": min(lons),
                "max_lon": max(lons),
                "min_lat": min(lats),
                "max_lat": max(lats)
            }
        
        # Save the combined map data
        with open('data/combined_routes_map.json', 'w') as f:
            json.dump(combined_map_data, f, indent=2)
        
        print("\nCombined map data saved to data/combined_routes_map.json")
        print("\nRoute Colors:")
        for criteria, style in combined_map_data["styles"].items():
            print(f"{criteria}: {style['route_color']}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 