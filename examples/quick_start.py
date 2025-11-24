"""
Quick Start Example for APEX Simulator
Run this to see a simplified demo
"""

import sys
sys.path.append('src')

from apex_simulator import APEXSimulator, SimulationConfig, ControllerType
import pandas as pd

def main():
    print("="*70)
    print("APEX Quick Start Example")
    print("="*70)
    
    # Create simple configuration
    config = SimulationConfig(
        events_per_config=100,  # Fewer events for quick demo
        seed=42
    )
    
    # Create simulator
    print("\n1. Creating simulator...")
    simulator = APEXSimulator(config)
    
    # Generate events
    print("\n2. Generating events...")
    events = simulator.generate_events()
    print(f"   Generated {len(events)} events")
    
    # Run comparison
    print("\n3. Running comparison...")
    results = simulator.run_comparison()
    
    # Print results
    print("\n4. Results:")
    print("-"*70)
    print(f"{'Controller':<15} {'p99 Latency (μs)':<20} {'Improvement'}")
    print("-"*70)
    
    baseline = results['no_control']['p99_latency_us']
    
    for controller in ['no_control', 'dibs', 'ecn', 'hpcc', 'apex']:
        if controller in results:
            p99 = results[controller]['p99_latency_us']
            improvement = ((baseline - p99) / baseline * 100) if controller != 'no_control' else 0
            
            imp_str = "baseline" if controller == 'no_control' else f"+{improvement:.1f}%"
            print(f"{controller:<15} {p99:<20.0f} {imp_str}")
    
    print("-"*70)
    
    # Export data
    print("\n5. Exporting data...")
    simulator.export_data()
    
    print("\n✅ Quick demo complete!")
    print("📊 Check data/ directory for CSV files")
    print("🚀 Run 'python src/apex_simulator.py' for full simulation")

if __name__ == "__main__":
    main()
