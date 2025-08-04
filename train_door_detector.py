"""
Door Detection Training Script
Usage: python train_door_detector.py
"""

import torch
from ultralytics import YOLO
import pandas as pd
import os

def main(): 
    # Load model
    model = YOLO('./models/weights/best_door_model.pt')
    
    results = model.train(
        data='./data.yml',
        epochs=200,
        imgsz=640,
        batch=24,
        lr0=0.0005,        # Fine-tuning learning rate
        warmup_epochs=10,
        
        # Optimized augmentation
        augment=True,
        hsv_h=0.015,
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=3.0,
        translate=0.15,
        scale=0.25,
        shear=1.0,
        perspective=0.0002,
        flipud=0.0,
        fliplr=0.5,
        mosaic=0.8,
        mixup=0.1,
        
        # Training settings
        cache=True,
        device=0,
        patience=25,
        save_period=25,    # Save checkpoints
        
        project='./merged_door_model',
        name='combined_optimized',
        verbose=True
    )
    
    # Print final metrics
    results_dir = './merged_door_model/combined_optimized'
    print(f"\nFinal Metrics:")
    
    if os.path.exists(f'{results_dir}/results.csv'):
        results_df = pd.read_csv(f'{results_dir}/results.csv')
        final_metrics = results_df.iloc[-1]  # Last epoch
        print(f"mAP50: {final_metrics.get('metrics/mAP50(B)', 'N/A'):.3f}")
        print(f"mAP50-95: {final_metrics.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
        print(f"Precision: {final_metrics.get('metrics/precision(B)', 'N/A'):.3f}")
        print(f"Recall: {final_metrics.get('metrics/recall(B)', 'N/A'):.3f}")
    else:
        print(f"Results file not found at {results_dir}/results.csv")
        print("Training may still be in progress or failed.")

if __name__ == "__main__":
    main()