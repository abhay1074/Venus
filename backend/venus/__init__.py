"""Venus AI — explainable diabetic-retinopathy screening pipeline (SIH26038).

One image enters, one annotated report leaves. Each stage is a plain function
with a typed dict in and out so it can be tested and timed on its own:

    stage0_gate      modality gate, field-of-view crop, quality score, enhancement
    stage1_segment   optic disc, fovea, vessels, lesions, ETDRS quadrants
    stage2_grade     CNN grader + ICDR rule grader + fusion + calibrated threshold
    stage3_explain   Grad-CAM, lesion overlay, attention agreement, PDF/JSON report
    stage4_simulate  district screening programme (discrete-event), sweep, Pareto
    stage5_schedule  priority tiers, slot allocation with ageing, SQLite records
    pipeline         screen_image(bytes) -> result, the end-to-end entry point
"""

__version__ = "1.0.0"
