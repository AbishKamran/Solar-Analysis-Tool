#!/usr/bin/env python3
"""
Solar Rooftop Analysis Tool
AI-powered rooftop analysis for solar installation potential assessment

Author: Solar Industry AI Assistant
Version: 1.1.0 - Fixed JSON parsing and error handling
"""

import streamlit as st
import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import math
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import numpy as np
import re

# Configuration
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Solar Industry Constants
SOLAR_PANEL_SPECS = {
    "residential": {
        "monocrystalline": {
            "efficiency": 0.22,
            "power_rating": 400,  # watts
            "dimensions": (2.0, 1.0),  # meters (length, width)
            "cost_per_watt": 2.50,
            "warranty_years": 25,
            "degradation_rate": 0.005  # 0.5% per year
        },
        "polycrystalline": {
            "efficiency": 0.18,
            "power_rating": 320,
            "dimensions": (2.0, 1.0),
            "cost_per_watt": 2.20,
            "warranty_years": 25,
            "degradation_rate": 0.007
        },
        "thin_film": {
            "efficiency": 0.12,
            "power_rating": 200,
            "dimensions": (2.0, 1.0),
            "cost_per_watt": 1.80,
            "warranty_years": 20,
            "degradation_rate": 0.008
        }
    }
}

SYSTEM_COSTS = {
    "inverter_cost_per_watt": 0.40,
    "mounting_cost_per_watt": 0.30,
    "electrical_cost_per_watt": 0.25,
    "labor_cost_per_watt": 0.60,
    "permit_inspection": 1500,
    "design_engineering": 800
}

INCENTIVES = {
    "federal_tax_credit": 0.30,  # 30% ITC
    "state_rebate_per_watt": 0.50,  # Example state rebate
    "srec_annual_value": 300  # Solar Renewable Energy Credits annual value
}

class SolarAnalysisEngine:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def encode_image(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_data = buffer.getvalue()
        return base64.b64encode(image_data).decode('utf-8')
    
    def extract_json_from_text(self, text: str) -> Dict:
        """Extract JSON from text response with multiple fallback methods"""
        if not text or not text.strip():
            raise ValueError("Empty response received")
        
        # Method 1: Look for JSON between { and }
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Method 2: Look for JSON between ```json and ```
        json_block_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_block_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Method 3: Try to find and fix common JSON issues
        try:
            # Remove any text before first { and after last }
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                json_str = text[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass
        
        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")
    
    def analyze_rooftop_image(self, image: Image.Image) -> Dict:
        """Analyze rooftop image using Vision AI with improved error handling"""
        
        try:
            base64_image = self.encode_image(image)
        except Exception as e:
            st.error(f"Error encoding image: {str(e)}")
            return self._get_fallback_analysis()
        
        prompt = """
        You are an expert solar installer analyzing a rooftop satellite image for solar panel installation potential.
        
        Analyze this rooftop image and provide a detailed assessment in the following JSON format.
        IMPORTANT: Respond ONLY with valid JSON, no additional text or formatting.
        
        {
            "rooftop_analysis": {
                "total_roof_area_sqm": 150,
                "usable_area_sqm": 120,
                "roof_orientation": "South",
                "roof_tilt_degrees": 30,
                "shading_assessment": {
                    "trees": "Minimal",
                    "buildings": "None",
                    "other_obstacles": "Standard vents and chimney"
                },
                "roof_condition": "Good",
                "access_difficulty": "Moderate",
                "structural_concerns": "None observed"
            },
            "solar_suitability": {
                "overall_score": 8,
                "primary_factors": ["Good south-facing orientation", "Minimal shading", "Adequate roof area"],
                "recommended_panel_layout": "Array on main south-facing section",
                "estimated_panel_capacity": 60
            },
            "additional_notes": "Analysis based on visible rooftop features"
        }
        
        Replace the example values with your actual analysis. Use realistic estimates for the uploaded image.
        Consider standard residential solar panel size of 2m x 1m when estimating capacity.
        """
        
        payload = {
            "model": "openai/gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.3
        }
        
        try:
            with st.spinner("Analyzing rooftop with AI..."):
                response = requests.post(
                    OPENROUTER_BASE_URL, 
                    headers=self.headers, 
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                
                result = response.json()
                
                if 'choices' not in result or not result['choices']:
                    raise ValueError("Invalid API response structure")
                
                content = result['choices'][0]['message']['content']
                
                if not content:
                    raise ValueError("Empty content in API response")
                
                # Extract JSON from response
                return self.extract_json_from_text(content)
                
        except requests.exceptions.RequestException as e:
            st.error(f"API request failed: {str(e)}")
            st.info("Using fallback analysis. Check your API key and internet connection.")
            return self._get_fallback_analysis()
        
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse AI response as JSON: {str(e)}")
            st.info("Using fallback analysis. The AI response format was unexpected.")
            return self._get_fallback_analysis()
        
        except Exception as e:
            st.error(f"Unexpected error during analysis: {str(e)}")
            st.info("Using fallback analysis.")
            return self._get_fallback_analysis()
    
    def _get_fallback_analysis(self) -> Dict:
        """Fallback analysis when API fails"""
        return {
            "rooftop_analysis": {
                "total_roof_area_sqm": 150,
                "usable_area_sqm": 120,
                "roof_orientation": "South",
                "roof_tilt_degrees": 30,
                "shading_assessment": {
                    "trees": "Minimal",
                    "buildings": "None",
                    "other_obstacles": "Standard vents and chimney"
                },
                "roof_condition": "Good",
                "access_difficulty": "Moderate",
                "structural_concerns": "None observed"
            },
            "solar_suitability": {
                "overall_score": 7,
                "primary_factors": ["Good south-facing orientation", "Minimal shading", "Adequate roof area"],
                "recommended_panel_layout": "Array on main south-facing section",
                "estimated_panel_capacity": 60
            },
            "additional_notes": "Fallback analysis - upload an image and configure API key for AI-powered assessment"
        }

class SolarCalculator:
    @staticmethod
    def calculate_system_capacity(panel_count: int, panel_type: str = "monocrystalline") -> float:
        """Calculate total system capacity in kW"""
        panel_power = SOLAR_PANEL_SPECS["residential"][panel_type]["power_rating"]
        return (panel_count * panel_power) / 1000  # Convert to kW
    
    @staticmethod
    def estimate_annual_production(system_capacity_kw: float, location_factor: float = 1400) -> float:
        """Estimate annual energy production in kWh"""
        # location_factor is annual sun hours * efficiency factors for location
        return system_capacity_kw * location_factor
    
    @staticmethod
    def calculate_system_cost(system_capacity_kw: float, panel_type: str = "monocrystalline") -> Dict:
        """Calculate total system costs"""
        capacity_watts = system_capacity_kw * 1000
        
        panel_specs = SOLAR_PANEL_SPECS["residential"][panel_type]
        
        costs = {
            "panels": capacity_watts * panel_specs["cost_per_watt"],
            "inverter": capacity_watts * SYSTEM_COSTS["inverter_cost_per_watt"],
            "mounting": capacity_watts * SYSTEM_COSTS["mounting_cost_per_watt"],
            "electrical": capacity_watts * SYSTEM_COSTS["electrical_cost_per_watt"],
            "labor": capacity_watts * SYSTEM_COSTS["labor_cost_per_watt"],
            "permits": SYSTEM_COSTS["permit_inspection"],
            "design": SYSTEM_COSTS["design_engineering"]
        }
        
        costs["subtotal"] = sum(costs.values())
        costs["contingency"] = costs["subtotal"] * 0.10  # 10% contingency
        costs["total"] = costs["subtotal"] + costs["contingency"]
        
        return costs
    
    @staticmethod
    def calculate_incentives(system_cost: float, system_capacity_kw: float) -> Dict:
        """Calculate available incentives"""
        capacity_watts = system_capacity_kw * 1000
        
        incentives = {
            "federal_tax_credit": system_cost * INCENTIVES["federal_tax_credit"],
            "state_rebate": capacity_watts * INCENTIVES["state_rebate_per_watt"],
            "srec_10_year": INCENTIVES["srec_annual_value"] * 10
        }
        
        incentives["total"] = sum(incentives.values())
        return incentives
    
    @staticmethod
    def calculate_roi_analysis(system_cost: float, annual_production: float, 
                             electricity_rate: float = 0.12) -> Dict:
        """Calculate ROI and payback analysis"""
        annual_savings = annual_production * electricity_rate
        
        # Account for rate escalation and system degradation
        total_savings_25_years = 0
        current_production = annual_production
        current_rate = electricity_rate
        
        for year in range(1, 26):
            total_savings_25_years += current_production * current_rate
            current_production *= 0.995  # 0.5% annual degradation
            current_rate *= 1.02  # 2% annual rate increase
        
        simple_payback = system_cost / annual_savings if annual_savings > 0 else float('inf')
        
        return {
            "annual_savings": annual_savings,
            "simple_payback_years": simple_payback,
            "total_25_year_savings": total_savings_25_years,
            "net_25_year_benefit": total_savings_25_years - system_cost,
            "roi_percentage": ((total_savings_25_years - system_cost) / system_cost * 100) if system_cost > 0 else 0
        }

def create_solar_visualization(analysis_data: Dict, system_capacity: float) -> go.Figure:
    """Create interactive solar production visualization"""
    
    # Generate monthly production estimates
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Seasonal factors for solar production
    seasonal_factors = [0.6, 0.7, 0.85, 1.0, 1.15, 1.2, 1.25, 1.15, 1.0, 0.8, 0.65, 0.55]
    
    annual_production = SolarCalculator.estimate_annual_production(system_capacity)
    monthly_production = [annual_production * factor / 12 for factor in seasonal_factors]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=months,
        y=monthly_production,
        name='Monthly Production (kWh)',
        marker_color='gold'
    ))
    
    fig.update_layout(
        title='Estimated Monthly Solar Production',
        xaxis_title='Month',
        yaxis_title='Energy Production (kWh)',
        template='plotly_white'
    )
    
    return fig

def create_cost_breakdown_chart(costs: Dict) -> go.Figure:
    """Create cost breakdown pie chart"""
    
    labels = ['Panels', 'Inverter', 'Mounting', 'Electrical', 'Labor', 'Permits', 'Design', 'Contingency']
    values = [costs['panels'], costs['inverter'], costs['mounting'], 
              costs['electrical'], costs['labor'], costs['permits'], 
              costs['design'], costs['contingency']]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(title_text="System Cost Breakdown")
    
    return fig

def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    if not api_key or len(api_key) < 10:
        return False
    return True

def main():
    st.set_page_config(
        page_title="Solar Rooftop Analysis Tool",
        page_icon="☀️",
        layout="wide"
    )
    
    st.title("☀️ AI-Powered Solar Rooftop Analysis Tool")
    st.markdown("*Professional solar installation potential assessment using satellite imagery and AI*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            value=OPENROUTER_API_KEY,
            help="Get your API key from OpenRouter.ai"
        )
        
        # Validate API key
        if api_key and not validate_api_key(api_key):
            st.warning("⚠️ API key appears to be invalid")
        elif not api_key:
            st.info("💡 Add your OpenRouter API key for AI-powered analysis")
        
        st.header("System Parameters")
        electricity_rate = st.slider(
            "Electricity Rate ($/kWh)",
            min_value=0.08,
            max_value=0.30,
            value=0.12,
            step=0.01
        )
        
        panel_type = st.selectbox(
            "Panel Type",
            options=["monocrystalline", "polycrystalline", "thin_film"],
            index=0
        )
        
        location_factor = st.slider(
            "Location Solar Factor",
            min_value=1000,
            max_value=2000,
            value=1400,
            help="Annual sun hours × efficiency factors for your location"
        )
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📷 Upload Rooftop Image")
        uploaded_file = st.file_uploader(
            "Choose a satellite/aerial image of the rooftop",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear aerial or satellite image of the rooftop for analysis"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Rooftop Image", use_container_width=True)
            
            if st.button("🔍 Analyze Rooftop", type="primary"):
                analyzer = SolarAnalysisEngine(api_key if api_key else "")
                analysis = analyzer.analyze_rooftop_image(image)
                st.session_state.analysis = analysis
                st.rerun()
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis' in st.session_state:
            analysis = st.session_state.analysis
            
            # Display rooftop analysis
            roof_data = analysis['rooftop_analysis']
            solar_data = analysis['solar_suitability']
            
            # Key metrics
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                st.metric(
                    "Suitability Score",
                    f"{solar_data['overall_score']}/10",
                    delta=None
                )
            
            with col2b:
                st.metric(
                    "Usable Area",
                    f"{roof_data['usable_area_sqm']} m²",
                    delta=None
                )
            
            with col2c:
                st.metric(
                    "Est. Panel Count",
                    f"{solar_data['estimated_panel_capacity']}",
                    delta=None
                )
            
            # Detailed analysis
            with st.expander("📋 Detailed Rooftop Analysis", expanded=True):
                st.write(f"**Roof Orientation:** {roof_data['roof_orientation']}")
                st.write(f"**Roof Tilt:** {roof_data['roof_tilt_degrees']}°")
                st.write(f"**Roof Condition:** {roof_data['roof_condition']}")
                st.write(f"**Access Difficulty:** {roof_data['access_difficulty']}")
                
                st.write("**Shading Assessment:**")
                st.write(f"- Trees: {roof_data['shading_assessment']['trees']}")
                st.write(f"- Buildings: {roof_data['shading_assessment']['buildings']}")
                st.write(f"- Other obstacles: {roof_data['shading_assessment']['other_obstacles']}")
            
            # Calculate system specifications
            panel_count = solar_data['estimated_panel_capacity']
            system_capacity = SolarCalculator.calculate_system_capacity(panel_count, panel_type)
            annual_production = SolarCalculator.estimate_annual_production(system_capacity, location_factor)
            
            # Calculate costs and ROI
            system_costs = SolarCalculator.calculate_system_cost(system_capacity, panel_type)
            incentives = SolarCalculator.calculate_incentives(system_costs['total'], system_capacity)
            net_cost = system_costs['total'] - incentives['total']
            roi_analysis = SolarCalculator.calculate_roi_analysis(net_cost, annual_production, electricity_rate)
        else:
            st.info("Upload and analyze a rooftop image to see detailed results")
    
    # Financial Analysis Section
    if 'analysis' in st.session_state:
        st.header("💰 Financial Analysis")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            st.subheader("System Specifications")
            
            specs_data = {
                "Metric": [
                    "System Capacity",
                    "Number of Panels",
                    "Panel Type",
                    "Annual Production",
                    "CO₂ Offset (Annual)"
                ],
                "Value": [
                    f"{system_capacity:.1f} kW",
                    f"{panel_count} panels",
                    panel_type.title(),
                    f"{annual_production:,.0f} kWh",
                    f"{annual_production * 0.4:.1f} tons"
                ]
            }
            
            st.table(pd.DataFrame(specs_data))
            
            st.subheader("Cost Breakdown")
            st.plotly_chart(create_cost_breakdown_chart(system_costs), use_container_width=True)
        
        with col4:
            st.subheader("Financial Summary")
            
            financial_data = {
                "Item": [
                    "System Cost",
                    "Federal Tax Credit (30%)",
                    "State Rebates",
                    "Net Cost",
                    "Annual Savings",
                    "Payback Period",
                    "25-Year ROI"
                ],
                "Amount": [
                    f"${system_costs['total']:,.0f}",
                    f"-${incentives['federal_tax_credit']:,.0f}",
                    f"-${incentives['state_rebate']:,.0f}",
                    f"${net_cost:,.0f}",
                    f"${roi_analysis['annual_savings']:,.0f}",
                    f"{roi_analysis['simple_payback_years']:.1f} years",
                    f"{roi_analysis['roi_percentage']:.0f}%"
                ]
            }
            
            st.table(pd.DataFrame(financial_data))
            
            # ROI indicator
            if roi_analysis['roi_percentage'] > 100:
                st.success(f"✅ Excellent Investment: {roi_analysis['roi_percentage']:.0f}% ROI over 25 years")
            elif roi_analysis['roi_percentage'] > 50:
                st.success(f"✅ Good Investment: {roi_analysis['roi_percentage']:.0f}% ROI over 25 years")
            else:
                st.warning(f"⚠️ Marginal Investment: {roi_analysis['roi_percentage']:.0f}% ROI over 25 years")
        
        # Production visualization
        st.subheader("📈 Solar Production Forecast")
        production_chart = create_solar_visualization(analysis, system_capacity)
        st.plotly_chart(production_chart, use_container_width=True)
        
        # Recommendations
        st.header("💡 Recommendations")
        
        recommendations = []
        
        if solar_data['overall_score'] >= 8:
            recommendations.append("✅ **Highly Recommended**: Excellent solar potential with strong ROI")
        elif solar_data['overall_score'] >= 6:
            recommendations.append("✅ **Recommended**: Good solar potential with reasonable payback")
        else:
            recommendations.append("⚠️ **Consider Carefully**: Limited solar potential, evaluate alternatives")
        
        if roof_data['roof_orientation'] in ['South', 'Southeast', 'Southwest']:
            recommendations.append("✅ **Optimal Orientation**: Excellent sun exposure throughout the day")
        
        if roof_data['shading_assessment']['trees'] in ['Moderate', 'Significant']:
            recommendations.append("🌳 **Tree Management**: Consider tree trimming to reduce shading")
        
        if roi_analysis['simple_payback_years'] < 8:
            recommendations.append("💰 **Fast Payback**: System will pay for itself quickly")
        
        recommendations.append(f"🔧 **Recommended Setup**: {solar_data['recommended_panel_layout']}")
        
        for rec in recommendations:
            st.write(rec)
        
        # Next Steps
        st.header("🚀 Next Steps")
        st.write("""
        **For Homeowners:**
        1. Get quotes from 3-5 local solar installers
        2. Verify roof structural integrity with engineer
        3. Check local permitting requirements
        4. Review utility interconnection policies
        5. Schedule site assessment with chosen installer
        
        **For Solar Professionals:**
        1. Conduct detailed site survey
        2. Perform structural analysis
        3. Create detailed system design
        4. Provide accurate pricing proposal
        5. Handle permitting and installation
        """)

if __name__ == "__main__":
    main()