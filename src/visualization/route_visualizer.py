import folium
import folium.plugins as plugins
from folium import plugins as folium_plugins
import branca.colormap as cm
from typing import List
from models.co2_optimizer import CO2RouteSegment

class RouteVisualizer:
    def __init__(self):
        self.colormap = cm.LinearColormap(
            colors=['green', 'yellow', 'red'],
            vmin=0,
            vmax=100,  # Maximum emissions in kg CO2
            caption='CO2 Emissions (kg)'
        )
        self.map = None
    
    def create_map(self, route_segments: List[CO2RouteSegment]):
        """
        Create an interactive map showing the route with detailed road geometry
        """
        if not route_segments:
            raise ValueError("No route segments provided")
        
        # Calculate center point from first segment
        first_segment = route_segments[0]
        center_lat = (first_segment.start_coords[1] + first_segment.end_coords[1]) / 2
        center_lon = (first_segment.start_coords[0] + first_segment.end_coords[0]) / 2
        
        self.map = folium.Map(location=[center_lat, center_lon], zoom_start=8)
        
        # Track cumulative emissions and calculate total
        cumulative_emissions = 0
        total_emissions = sum(segment.emissions for segment in route_segments)
        
        # Add route segments to map
        for i, segment in enumerate(route_segments):
            # Update cumulative emissions
            cumulative_emissions += segment.emissions
            
            # Calculate percentages
            segment_percentage = (segment.emissions / total_emissions) * 100
            cumulative_percentage = (cumulative_emissions / total_emissions) * 100
            
            # Create detailed popup content
            popup_content = f"""
                <div style='font-family: Arial, sans-serif;'>
                    <h4 style='margin-bottom: 10px;'>{segment.start} → {segment.end}</h4>
                    <p><strong>Distance:</strong> {segment.distance:.1f} km</p>
                    <p><strong>Estimated Time:</strong> {segment.time:.1f} hours</p>
                    <p><strong>Segment CO2:</strong> {segment.emissions:.2f} kg ({segment_percentage:.1f}% of total)</p>
                    <p><strong>Cumulative CO2:</strong> {cumulative_emissions:.2f} kg ({cumulative_percentage:.1f}% of total)</p>
                    <p><strong>Traffic Factor:</strong> {segment.traffic_factor:.2f}x</p>
                    <p><strong>Weather Factor:</strong> {segment.weather_factor:.2f}x</p>
                    <p><strong>Elevation Gain:</strong> {segment.elevation_gain:.1f} m</p>
                    <p><strong>Speed Limit:</strong> {segment.speed_limit} km/h</p>
                </div>
            """
            
            # Add markers for start and end points with detailed popups
            folium.Marker(
                [segment.start_coords[1], segment.start_coords[0]],
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(self.map)
            
            # Add detailed route line with all coordinates
            route_coords = [[coord[1], coord[0]] for coord in segment.route_geometry]
            folium.PolyLine(
                locations=route_coords,
                weight=3,
                color='red',
                opacity=0.8,
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(self.map)
            
            # Add a line for the direct path (dashed) to compare with actual route
            folium.PolyLine(
                locations=[[segment.start_coords[1], segment.start_coords[0]],
                          [segment.end_coords[1], segment.end_coords[0]]],
                weight=1,
                color='blue',
                opacity=0.5,
                dash_array='5',
                popup=folium.Popup(popup_content, max_width=300)
            ).add_to(self.map)
            
            # Add percentage markers at the end of each segment
            percentage_color = 'red' if segment_percentage > 30 else 'orange' if segment_percentage > 15 else 'green'
            folium.Marker(
                [segment.end_coords[1], segment.end_coords[0]],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12pt; color: {percentage_color}; background-color: white; padding: 5px; border-radius: 5px; border: 1px solid {percentage_color};">'
                         f'{cumulative_percentage:.1f}%</div>'
                )
            ).add_to(self.map)
            
            # Add a final marker with total emissions at the end of the route
            if i == len(route_segments) - 1:  # Only for the last segment
                folium.Marker(
                    [segment.end_coords[1], segment.end_coords[0]],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 14pt; color: red; background-color: white; padding: 5px; border-radius: 5px; border: 2px solid red;">'
                             f'Total: {total_emissions:.2f} kg</div>'
                    )
                ).add_to(self.map)
        
        # Add a legend
        legend_html = '''
            <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 2px solid grey; border-radius: 5px;">
                <h4>Route Information</h4>
                <p>Red Line: Actual Route</p>
                <p>Blue Dashed Line: Direct Path</p>
                <p>Markers: Stops with Detailed Info</p>
                <p>Percentage Markers:</p>
                <p style="color: green;">● < 15%</p>
                <p style="color: orange;">● 15-30%</p>
                <p style="color: red;">● > 30%</p>
            </div>
        '''
        self.map.get_root().html.add_child(folium.Element(legend_html))
    
    def save_map(self, filename: str = "route_map.html"):
        """
        Save the map to an HTML file
        """
        if self.map is None:
            raise ValueError("No map has been created yet")
        self.map.save(filename)
    
    def create_route_map(self, route_segments: List[CO2RouteSegment], total_emissions: float) -> folium.Map:
        """
        Create an interactive map showing the route and emissions
        """
        # Create base map
        m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)  # Default to London
        
        # Add route segments
        for segment in route_segments:
            # Get coordinates for start and end points
            # Note: In a real implementation, you would need to geocode these addresses
            start_coords = [0, 0]  # Placeholder
            end_coords = [0, 0]    # Placeholder
            
            # Create line with color based on emissions
            color = self.colormap(segment.emissions)
            folium.PolyLine(
                locations=[start_coords, end_coords],
                color=color,
                weight=5,
                opacity=0.8,
                popup=f"Distance: {segment.distance:.2f} km<br>"
                      f"Emissions: {segment.emissions:.2f} kg CO2<br>"
                      f"Time: {segment.time:.2f} hours"
            ).add_to(m)
            
            # Add markers for start and end points
            folium.Marker(
                location=start_coords,
                popup=f"Start: {segment.start}<br>"
                      f"Emissions: {segment.emissions:.2f} kg CO2"
            ).add_to(m)
            
            folium.Marker(
                location=end_coords,
                popup=f"End: {segment.end}<br>"
                      f"Emissions: {segment.emissions:.2f} kg CO2"
            ).add_to(m)
        
        # Add color scale
        self.colormap.add_to(m)
        
        # Add total emissions info
        folium.Marker(
            location=[0, 0],  # Placeholder
            icon=folium.DivIcon(
                html=f'<div style="font-size: 16pt; color: black;">'
                     f'Total Emissions: {total_emissions:.2f} kg CO2</div>'
            )
        ).add_to(m)
        
        return m
    
    def create_emissions_chart(self, route_segments: List[CO2RouteSegment]):
        """
        Create a bar chart showing emissions for each segment
        """
        import matplotlib.pyplot as plt
        
        segments = [f"{seg.start} to {seg.end}" for seg in route_segments]
        emissions = [seg.emissions for seg in route_segments]
        
        plt.figure(figsize=(12, 6))
        plt.bar(segments, emissions, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('CO2 Emissions (kg)')
        plt.title('Emissions by Route Segment')
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_scenario_visualization(self, scenarios):
        """
        Create visualization for future scenarios
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        for i, scenario in enumerate(scenarios):
            plt.plot(scenario, label=f'Scenario {i+1}')
        
        plt.xlabel('Route Segment')
        plt.ylabel('Normalized Value')
        plt.title('Future Route Scenarios')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf() 