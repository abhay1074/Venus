function result = screenImage(imagePath, models, opts)
%SCREENIMAGE  End-to-end: one image in, one annotated result out.
%   result = drscreen.screenImage(imagePath, models)
%   result = drscreen.screenImage(imagePath, models, struct('intake', intake, 'tta', false, 'report', true))
%
%   Runs Stage 0 -> 1 -> 2 -> 3 -> tier, writes the PDF/JSON report when
%   opts.report is true, and returns the result struct with the same fields
%   as the Python pipeline (stage0, stage1, stage2, stage3, stage5, timingMs,
%   modelVersion, calibrationFingerprint, recommendation).

    if nargin < 3, opts = struct(); end
    intake = getfieldOr(opts, 'intake', struct());
    tta = getfieldOr(opts, 'tta', false);
    writeReport = getfieldOr(opts, 'report', true);
    tAll = tic;
    point = models.point;
    sessionId = ['VS-' upper(char(java.util.UUID.randomUUID()))];
    sessionId = sessionId(1:13);
    capturedAt = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssZ'));
    image = imread(imagePath);
    if size(image, 3) == 1, image = repmat(image, [1 1 3]); end

    s0 = drscreen.gate(image, models);
    if ~s0.accepted
        t = drscreen.tier([], intake);
        result = struct('sessionId', sessionId, 'capturedAt', capturedAt, 'modelVersion', point.modelVersion, ...
            'calibrationFingerprint', point.calibrationFingerprint, 'accepted', false, 'stopReason', s0.stopReason, ...
            'stage0', rmfield(s0, intersect(fieldnames(s0), {'image', 'mask', 'original'})), 'stage5', t, ...
            'recommendation', recommendation('P0'), 'timingMs', struct('stage0', s0.elapsedMs, 'total', round(1000 * toc(tAll))));
        return;
    end
    s1 = drscreen.segment(s0.image, s0.mask, models, s0.original, image);
    t2 = tic;
    cnn = drscreen.gradeCNN(s0, models, point, tta);
    rule = drscreen.gradeRule(s1, cnn.nvProbability);
    fusion = drscreen.fuse(cnn, rule, point, models.policy);
    s2 = struct('cnn', cnn, 'rule', rule, 'fusion', fusion, 'elapsedMs', round(1000 * toc(t2)));
    s3 = drscreen.explain(s0, s1, s2, models);
    if s3.attentionAgreement.flag
        s2.fusion.flagReasons{end+1} = sprintf('attention agreement %.2f: Grad-CAM mass is not on the detected lesions', s3.attentionAgreement.score);
        s2.fusion.flagForReview = true;
    end
    partial = struct('accepted', true, 'stage2', s2);
    t5 = drscreen.tier(partial, intake);
    s2.cnn = rmfield(s2.cnn, 'input');

    result = struct('sessionId', sessionId, 'capturedAt', capturedAt, 'modelVersion', point.modelVersion, ...
        'calibrationFingerprint', point.calibrationFingerprint, 'accepted', true, 'stopReason', '', ...
        'stage0', rmfield(s0, {'image', 'mask', 'original'}), 'stage1', s1, 'stage2', s2, 'stage3', s3, 'stage5', t5, ...
        'recommendation', recommendation(t5.tier), ...
        'timingMs', struct('stage0', s0.elapsedMs, 'stage1', s1.elapsedMs, 'stage2', s2.elapsedMs, ...
            'stage3', s3.elapsedMs, 'gradcam', s3.gradcamMs, 'total', round(1000 * toc(tAll))));
    if writeReport
        result.report = drscreen.report(result);
    end
end

function text = recommendation(tierCode)
    switch tierCode
        case 'P0', text = 'Image not gradable. Retake on the same visit following the reason shown to the operator.';
        case 'P1', text = 'Urgent: proliferative-stage evidence or sight-threatening symptoms. Book an in-person ophthalmology appointment at the district hospital within 7 days.';
        case 'P2', text = 'Referable diabetic retinopathy. Book an ophthalmology or tele-review appointment within 30 days. This is a screening result read alongside the evidence on this page, not a diagnosis.';
        case 'P3', text = 'The two graders or the confidence band disagree. A human reader reviews this image within 3 days with both grades and the evidence attached, then the case is re-tiered.';
        otherwise, text = 'No referable diabetic retinopathy found on this image. Routine recall; sooner if symptoms appear. A negative screen does not exclude disease.';
    end
end

function v = getfieldOr(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
