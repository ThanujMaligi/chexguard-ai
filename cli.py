"""
CheXGuard AI — Command Line Interface (CLI).
Enables command-line inference, OOD validation, batch testing, and benchmark auditing.
"""

import os
import sys
import argparse
import json
import torch
from PIL import Image

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from utils.image_processing import preprocess_radiograph
from utils.report_generator import compute_triage_priority, generate_markdown_report, generate_pdf_report
from models.backbone import CheXGuardNet, build_calibrated_model, TARGET_PATHOLOGIES, URGENCY_WEIGHTS
from models.gradcam import GradCAM
from models.uncertainty import MonteCarloUncertaintyEngine
from models.ood_detector import OODDetector


def run_predict(args):
    """Executes single-image diagnostic inference, uncertainty estimation, and triage report."""
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print(f"\n=======================================================")
    print(f"   CheXGuard AI: Clinical Radiograph Diagnostic Engine")
    print(f"=======================================================\n")
    print(f"[1/4] Loading input image: {image_path}")
    raw_img = Image.open(image_path)

    # 1. Safety Guardrail: OOD Detection
    ood_detector = OODDetector()
    ood_res = ood_detector.inspect_image(raw_img)
    print(f"[2/4] Safety Guardrail Check:")
    print(f"      Conforming Radiograph: {ood_res['is_valid']}")
    print(f"      OOD Risk Score:        {ood_res['ood_risk_score']} / 1.00")
    print(f"      Safety Message:        {ood_res['reason']}")

    if not ood_res["is_valid"] and not args.force:
        print("\n[ALERT] Input rejected by Responsible AI guardrails.")
        print("        To force inference on non-conforming images, re-run with --force.")
        sys.exit(2)

    # 2. Preprocess & Model
    tensor, display_img = preprocess_radiograph(raw_img)
    model = build_calibrated_model()
    uncertainty_engine = MonteCarloUncertaintyEngine(model, TARGET_PATHOLOGIES)

    # 3. Monte Carlo Bayesian Inference
    print(f"[3/4] Running Bayesian Monte Carlo Dropout ({args.mc_samples} iterations)...")
    uq_res = uncertainty_engine.sample_predictions(tensor, num_iterations=args.mc_samples)
    predictions = uq_res["pathology_metrics"]

    # 4. Triage Priority Assessment
    triage_info = compute_triage_priority(predictions, URGENCY_WEIGHTS)
    print(f"[4/4] Triage Priority: {triage_info['priority_tier']}")
    print(f"      Primary Concern: {triage_info['primary_concern']}")
    print(f"      Urgency Index:   {triage_info['urgency_score']:.2f}")

    print("\n--------------------------------------------------------------------------------")
    print(f"{'Thoracic Pathology':<20} | {'Probability':<12} | {'95% CI':<18} | {'Reliability'}")
    print("--------------------------------------------------------------------------------")
    for name in TARGET_PATHOLOGIES:
        p = predictions[name]
        prob = f"{p['mean_probability']*100:.1f}%"
        ci = f"[{p['ci_95_lower']*100:.1f}%, {p['ci_95_upper']*100:.1f}%]"
        badge = p['badge']
        print(f"{name:<20} | {prob:<12} | {ci:<18} | {badge}")
    print("--------------------------------------------------------------------------------")

    # Generate Reports
    scan_id = os.path.basename(image_path).replace(".png", "").replace(".jpg", "").upper()
    if args.pdf_output:
        pdf_path = generate_pdf_report(args.pdf_output, scan_id, predictions, triage_info, ood_res)
        print(f"\n[+] PDF Clinical Triage Report successfully saved to: {pdf_path}")

    if args.markdown_output:
        md_text = generate_markdown_report(scan_id, predictions, triage_info, ood_res)
        with open(args.markdown_output, "w") as f:
            f.write(md_text)
        print(f"[+] Markdown Summary saved to: {args.markdown_output}")


def run_validate_ood(args):
    """Runs OOD safety check on an image."""
    image_path = args.image
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    raw_img = Image.open(image_path)
    detector = OODDetector()
    res = detector.inspect_image(raw_img)
    print(json.dumps(res, indent=2))


def run_evaluate_samples(args):
    """Evaluates all sample cases in data/sample_cases."""
    sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_cases")
    if not os.path.exists(sample_dir):
        print(f"Sample directory not found: {sample_dir}")
        sys.exit(1)

    files = [f for f in os.listdir(sample_dir) if f.endswith(('.png', '.jpg'))]
    files.sort()
    print(f"\nEvaluating {len(files)} sample cases in {sample_dir}...\n")

    model = build_calibrated_model()
    detector = OODDetector()
    uq_engine = MonteCarloUncertaintyEngine(model, TARGET_PATHOLOGIES)

    for fname in files:
        fpath = os.path.join(sample_dir, fname)
        img = Image.open(fpath)
        ood = detector.inspect_image(img)
        print(f"File: {fname:<25} | Valid Radiograph: {str(ood['is_valid']):<6} | OOD Risk: {ood['ood_risk_score']:<5} | Note: {ood['reason'][:40]}...")


def run_show_benchmarks(args):
    """Prints the benchmark evaluation metrics."""
    bench_file = os.path.join(os.path.dirname(__file__), "benchmarks", "benchmark_results.json")
    if not os.path.exists(bench_file):
        print("Benchmarks not found. Run benchmarks/generate_benchmarks.py first.")
        sys.exit(1)

    with open(bench_file, "r") as f:
        data = json.load(f)

    print("\n=======================================================")
    print(f"   CheXGuard AI — Published Evaluation Benchmarks")
    print("=======================================================")
    print(f"Cohort:       {data['dataset_name']}")
    print(f"Mean Macro AUC: {data['aggregate_metrics']['mean_macro_auc']:.4f}")
    print(f"Latency CPU:    {data['aggregate_metrics']['latency_mean_ms']:.2f} ms")
    print(f"Throughput:     {data['aggregate_metrics']['throughput_fps']:.1f} FPS\n")
    print("Pathology Breakdown:")
    for name, stats in data["pathology_benchmarks"].items():
        print(f"  • {name:<18} | ROC-AUC: {stats['roc_auc']:.4f} | Recall: {stats['sensitivity_recall']:.4f} | F1: {stats['f1_score']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="CheXGuard AI CLI — Medical Image Diagnostic Assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Run diagnostic inference on a chest radiograph")
    pred_parser.add_argument("--image", required=True, help="Path to input image")
    pred_parser.add_argument("--mc-samples", type=int, default=15, help="Number of Monte Carlo Dropout iterations")
    pred_parser.add_argument("--pdf-output", help="Path to save clinical PDF triage report")
    pred_parser.add_argument("--markdown-output", help="Path to save markdown summary")
    pred_parser.add_argument("--force", action="store_true", help="Force prediction even if flagged as OOD")

    # Validate OOD
    ood_parser = subparsers.add_parser("validate-ood", help="Check if an image is a valid radiograph")
    ood_parser.add_argument("--image", required=True, help="Path to input image")

    # Evaluate Samples
    subparsers.add_parser("evaluate-samples", help="Batch evaluate all built-in sample cases")

    # Show Benchmarks
    subparsers.add_parser("show-benchmarks", help="Display verified model benchmark metrics")

    args = parser.parse_args()
    if args.command == "predict":
        run_predict(args)
    elif args.command == "validate-ood":
        run_validate_ood(args)
    elif args.command == "evaluate-samples":
        run_evaluate_samples(args)
    elif args.command == "show-benchmarks":
        run_show_benchmarks(args)


if __name__ == "__main__":
    main()
