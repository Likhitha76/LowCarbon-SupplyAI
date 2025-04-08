import folium
from folium import plugins
import branca.colormap as cm
from typing import List
from ..models.route_optimizer import RouteSegment

class RouteVisualizer:
    def __init__(self):
        self.colormap = cm.LinearColormap(
            ['green', 'yellow', 'red'],
            vmin=0,
            vmax=100,  # Maximum emissions in kg CO2
            caption='CO2 Emissions (kg)'
        )
    
    def create_route_map(self, route_segments: List[RouteSegment], total_emissions: float) -> folium.Map:
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
    
    def create_emissions_chart(self, route_segments: List[RouteSegment]):
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