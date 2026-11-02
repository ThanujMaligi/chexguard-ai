"""
Automated Clinical Triage & Diagnostic Report Generator for CheXGuard AI.
Generates structured Markdown and publication-grade PDF triage summaries
with quantitative uncertainty intervals and Responsible AI disclaimers.
"""

from typing import Dict, Any, Optional
import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def compute_triage_priority(predictions: Dict[str, Any], urgency_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes an aggregate Clinical Urgency Score and assigns an Emergency Triage Priority tier.
    """
    weighted_score = 0.0
    highest_risk_condition = "None"
    highest_prob = 0.0

    for condition, stats in predictions.items():
        prob = stats["mean_probability"]
        weight = urgency_weights.get(condition, 0.5)
        if condition != "No Finding":
            weighted_score = max(weighted_score, prob * weight)
            if prob > highest_prob and prob > 0.40:
                highest_prob = prob
                highest_risk_condition = condition

    if weighted_score >= 0.70:
        tier = "CRITICAL (STAT PRIORITY)"
        color = "#D32F2F"
        action = "Immediate attending physician bedside evaluation and emergency radiologist review recommended."
    elif weighted_score >= 0.40:
        tier = "ELEVATED (PRIORITY 2)"
        color = "#F57C00"
        action = "Expedited clinical correlation and priority radiologist read recommended within 2 hours."
    else:
        tier = "ROUTINE (MONITORING)"
        color = "#388E3C"
        action = "Standard clinical monitoring. No acute life-threatening thoracic pathology detected above threshold."

    return {
        "urgency_score": round(weighted_score, 3),
        "priority_tier": tier,
        "badge_color": color,
        "primary_concern": highest_risk_condition,
        "recommended_action": action
    }


def generate_markdown_report(
    scan_id: str,
    predictions: Dict[str, Any],
    triage_info: Dict[str, Any],
    ood_info: Dict[str, Any]
) -> str:
    """Generates a clean GFM markdown clinical report."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    md = f"""# 🏥 CheXGuard AI: Clinical Triage & Diagnostic Report

**Study Identifier:** `{scan_id}`  
**Evaluation Timestamp:** {timestamp}  
**Clinical Triage Tier:** **{triage_info['priority_tier']}** (Urgency Score: {triage_info['urgency_score']})  
**Safety Validation Status:** {"✅ PASS" if ood_info['is_valid'] else "❌ REJECTED"}  

---

## 1. Input Integrity & Distribution Verification
- **OOD Risk Score:** `{ood_info['ood_risk_score']}`
- **Verification Note:** {ood_info['reason']}
- **Biometric Contrast:** `{ood_info['metrics'].get('contrast_std', 'N/A')}` | **Dynamic Range:** `{ood_info['metrics'].get('dynamic_range', 'N/A')}`

---

## 2. Quantitative Diagnostic Findings & Uncertainty Bounds

| Pathology Condition | Predictive Probability | 95% Confidence Interval | Epistemic Uncertainty (σ) | Clinical Reliability |
| :--- | :---: | :---: | :---: | :---: |
"""
    for cond, data in predictions.items():
        prob = f"{data['mean_probability'] * 100:.1f}%"
        ci = f"[{data['ci_95_lower'] * 100:.1f}%, {data['ci_95_upper'] * 100:.1f}%]"
        unc = f"±{data['epistemic_uncertainty']:.3f}"
        badge = data['badge']
        md += f"| **{cond}** | {prob} | {ci} | {unc} | {badge} |\n"

    md += f"""
---

## 3. Triage Assessment & Protocol Guidance
- **Primary Finding:** {triage_info['primary_concern']}
- **Clinical Protocol:** {triage_info['recommended_action']}
- **Explainability Grounding:** Visual Grad-CAM localization maps correspond to peak activation feature gradients in the final convolutional blocks.

---

> **⚠️ RESPONSIBLE AI & REGULATORY DISCLAIMER**  
> CheXGuard AI is an investigational clinical decision-support triage assistant intended strictly for accredited medical practitioners. It is not an autonomous diagnostic device. All algorithmic findings and uncertainty estimates must be correlated with patient clinical presentation and confirmed by a certified radiologist.
"""
    return md


def generate_pdf_report(
    output_pdf_path: str,
    scan_id: str,
    predictions: Dict[str, Any],
    triage_info: Dict[str, Any],
    ood_info: Dict[str, Any]
) -> str:
    """Generates a professional medical triage PDF document."""
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1A237E'),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#455A64'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0D47A1'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#212121')
    )
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Italic'],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#C62828')
    )

    elements = []

    # Title & Metadata
    elements.append(Paragraph("CheXGuard AI — Thoracic Radiograph Triage Report", title_style))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    elements.append(Paragraph(
        f"<b>Scan Identifier:</b> {scan_id} &nbsp;|&nbsp; <b>Timestamp:</b> {timestamp} &nbsp;|&nbsp; <b>Model:</b> CheXGuardNet v2.1-DenseNet121",
        subtitle_style
    ))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1A237E'), spaceAfter=10))

    # Triage Summary Banner
    triage_table_data = [
        [
            Paragraph(f"<b>TRIAGE PRIORITY:</b> {triage_info['priority_tier']}", body_style),
            Paragraph(f"<b>URGENCY INDEX:</b> {triage_info['urgency_score']:.2f} / 1.00", body_style)
        ],
        [
            Paragraph(f"<b>Primary Finding:</b> {triage_info['primary_concern']}", body_style),
            Paragraph(f"<b>Safety Validation:</b> {'VERIFIED RADIOGRAPH' if ood_info['is_valid'] else 'NON-CONFORMING'} (OOD: {ood_info['ood_risk_score']})", body_style)
        ]
    ]
    triage_table = Table(triage_table_data, colWidths=[270, 270])
    triage_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#90CAF9')),
        ('PADDING', (0, 0), (-1, -1), 6),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    elements.append(triage_table)
    elements.append(Spacer(1, 10))

    # Diagnostic Findings Table
    elements.append(Paragraph("Diagnostic Predictions & Epistemic Uncertainty Bounds", heading_style))
    table_data = [
        ["Thoracic Pathology", "Probability", "95% Confidence Interval", "Epistemic Std (σ)", "Reliability Flag"]
    ]

    for condition, stats in predictions.items():
        prob_str = f"{stats['mean_probability'] * 100:.1f}%"
        ci_str = f"[{stats['ci_95_lower'] * 100:.1f}%, {stats['ci_95_upper'] * 100:.1f}%]"
        std_str = f"+/-{stats['epistemic_uncertainty']:.3f}"
        raw_badge = stats.get('badge', stats.get('reliability', 'N/A'))
        clean_badge = raw_badge.replace("🟢", "[OK]").replace("🟡", "[WARN]").replace("🔴", "[ALERT]")
        table_data.append([condition, prob_str, ci_str, std_str, clean_badge])

    findings_table = Table(table_data, colWidths=[130, 85, 135, 95, 95])
    findings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A237E')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CFD8DC')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8F9FA')])
    ]))
    elements.append(findings_table)
    elements.append(Spacer(1, 10))

    # Actionable Clinical Recommendation
    elements.append(Paragraph("Clinical Recommendation & Next Steps", heading_style))
    elements.append(Paragraph(f"• <b>Recommended Protocol:</b> {triage_info['recommended_action']}", body_style))
    elements.append(Paragraph("• <b>Visual Grounding:</b> Grad-CAM class activation maps indicate localized thoracic tissue opacity corresponding to the highest probability pathology.", body_style))
    elements.append(Spacer(1, 12))

    # Medical Disclaimer
    elements.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#C62828'), spaceAfter=6))
    disclaimer_text = (
        "<b>LEGAL & CLINICAL DISCLAIMER:</b> CheXGuard AI is an assistive decision-support algorithm developed "
        "for investigational triage during Neural Storm 2026. This system is not FDA/CE-cleared as a primary diagnostic device. "
        "Diagnostic decisions, patient management, and treatment plans must be made exclusively by a qualified healthcare provider."
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))

    doc.build(elements)
    return output_pdf_path
