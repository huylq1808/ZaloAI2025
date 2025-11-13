
---

## 🎓 **COMPLETE WORKFLOW SUMMARY**

### **Training Pipeline:**

```bash
# 1. Prepare data
python tracking/prepare_data.py --data_dir raw_data --output_dir data/train

# 2. Train model
python tracking/train_fewshot.py \
    --config configs/train_config.yaml \
    --device cuda

# 3. Monitor training
tensorboard --logdir checkpoints/fewshot_yolo/logs
``` 

### inference 

``` bash 
# 1. Run inference
python tracking/inference.py \
    --checkpoint checkpoints/fewshot_yolo/best.pt \
    --test_dir data/test/samples \
    --output predictions.json \
    --save_vis --vis_dir visualizations

# 2. Evaluate results
python tracking/evaluate.py \
    --predictions predictions.json \
    --ground_truth data/test/annotations/annotations.json \
    --output evaluation_results.json

# 3. Visualize results
python tools/visualize_predictions.py report \
    --eval_results evaluation_results.json \
    --output evaluation_report.html
```