import os
import sys
from dotenv import load_dotenv
from models.co2_optimizer import CO2Optimizer
from visualization.route_visualizer import RouteVisualizer

def main():
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    ors_api_key = os.getenv('OPENROUTESERVICE_API_KEY')
    weather_api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not ors_api_key or not weather_api_key:
        print("Error: API keys not found in .env file")
        print("Please add your API keys to your .env file:")
        print("OPENROUTESERVICE_API_KEY=your_key_here")
        print("OPENWEATHER_API_KEY=your_key_here")
        return
    
    try:
        # Initialize CO2 optimizer
        optimizer = CO2Optimizer(ors_api_key, weather_api_key)
        
        # Route parameters for Ireland journey
        origin = "Dublin, Ireland"
        destination = "Galway, Ireland"
        stops = [
            "Athlone, Ireland",  # Historic town in the heart of Ireland
            "Tullamore, Ireland",  # Famous for its whiskey distillery
            "Mullingar, Ireland"  # Vibrant midlands town
        ]
        vehicle_type = "van"  # Using a van instead of a truck for Irish roads
        load_weight = 500  # kg (medium load for deliveries)
        
        print(f"\nOptimizing route from {origin} to {destination}...")
        print(f"With stops at: {', '.join(stops)}")
        
        # Optimize route with real-time data
        route_segments, total_emissions = optimizer.optimize_route(
            origin=origin,
            destination=destination,
            stops=stops,
            vehicle_type=vehicle_type,
            load_weight=load_weight,
            time_window_start="09:00",
            time_window_end="17:00"
        )
        
        # Print results
        print(f"\nOptimized Route from {origin} to {destination}")
        print(f"Total CO2 Emissions: {total_emissions:.2f} kg")
        print("\nRoute Segments:")
        for segment in route_segments:
            print(f"\n{segment.start} -> {segment.end}")
            print(f"Distance: {segment.distance:.2f} km")
            print(f"Emissions: {segment.emissions:.2f} kg CO2")
            print(f"Estimated Time: {segment.time:.2f} hours")
            print(f"Traffic Factor: {segment.traffic_factor:.2f}")
            print(f"Weather Factor: {segment.weather_factor:.2f}")
            print(f"Elevation Gain: {segment.elevation_gain:.1f} m")
            print(f"Speed Limit: {segment.speed_limit} km/h")
        
        # Save route data for future training
        optimizer.save_route_data(route_segments)
        
        # Visualize the route
        print("\nCreating route visualization...")
        visualizer = RouteVisualizer()
        visualizer.create_map(route_segments)
        visualizer.save_map("route_map.html")
        print("Route map saved as 'route_map.html'")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nPlease check that:")
        print("1. Your API keys are valid")
        print("2. You have an active internet connection")
        print("3. The locations provided are valid")
        sys.exit(1)

if __name__ == "__main__":
    main() 