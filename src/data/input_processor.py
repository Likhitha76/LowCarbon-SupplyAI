import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RouteInput:
    origin: str
    destination: str
    vehicle_type: str
    load_weight: float
    time_window_start: datetime
    time_window_end: datetime
    stops: Optional[List[str]] = None
    priority: str = "emissions"  # emissions, time, or cost
    weather_conditions: Optional[Dict] = None
    traffic_conditions: Optional[Dict] = None

class InputProcessor:
    def __init__(self):
        self.vehicle_emission_factors = {
            "truck": 0.15,  # kg CO2 per km
            "van": 0.12,
            "car": 0.08
        }
        
    def process_input(self, input_data: Dict) -> RouteInput:
        """
        Process raw input data into structured RouteInput object
        """
        try:
            route_input = RouteInput(
                origin=input_data['origin'],
                destination=input_data['destination'],
                vehicle_type=input_data['vehicle_type'],
                load_weight=float(input_data['load_weight']),
                time_window_start=datetime.strptime(input_data['time_window_start'], '%Y-%m-%d %H:%M'),
                time_window_end=datetime.strptime(input_data['time_window_end'], '%Y-%m-%d %H:%M'),
                stops=input_data.get('stops', []),
                priority=input_data.get('priority', 'emissions'),
                weather_conditions=input_data.get('weather_conditions'),
                traffic_conditions=input_data.get('traffic_conditions')
            )
            return route_input
        except Exception as e:
            raise ValueError(f"Error processing input data: {str(e)}")

    def validate_input(self, route_input: RouteInput) -> bool:
        """
        Validate the processed input data
        """
        if route_input.vehicle_type not in self.vehicle_emission_factors:
            raise ValueError(f"Invalid vehicle type: {route_input.vehicle_type}")
        
        if route_input.load_weight <= 0:
            raise ValueError("Load weight must be positive")
        
        if route_input.time_window_start >= route_input.time_window_end:
            raise ValueError("Time window end must be after start")
        
        return True

    def get_emission_factor(self, vehicle_type: str) -> float:
        """
        Get the emission factor for a given vehicle type
        """
        return self.vehicle_emission_factors.get(vehicle_type, 0.15)  # Default to truck if unknown 