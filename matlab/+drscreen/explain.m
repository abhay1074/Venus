function s3 = explain(s0, s1, s2, models)
%EXPLAIN  Stage 3: Grad-CAM, overlays, attention agreement.
%   s3 = drscreen.explain(s0, s1, s2, models)
%   Grad-CAM (Deep Learning Toolbox gradCAM) on the grader for the referable
%   head (output 2, P(grade >= 2)) and the predicted grade's threshold; the
%   heatmap is clipped to the FOV and overlaid at 40 % opacity. Lesion
%   outlines are drawn on the ORIGINAL frame (MA red, HE dark red, EX yellow,
%   SE white), the disc and fovea in cyan. Attention agreement = share of
%   Grad-CAM mass inside a lesion neighbourhood (radius 20 px, one CAM cell),
%   with the chance level (neighbourhood share of the FOV) and the lift.
%   A referable call whose heat share is below the validation-chosen floor
%   (config/review_policy.json, 0.20; else 0.15 AND no better than chance)
%   is flagged. Mirrors backend.venus.stage3_explain.attention_agreement.

    t0 = tic;
    mask = s0.mask; original = s0.original;
    grade = s2.cnn.grade;
    gradeTarget = max(grade, 1);              % ordinal output index (1-based): grade>=k -> k
    heatRef = camFor(models, s2.cnn.input, 2, mask);
    if gradeTarget == 2
        heatGrade = heatRef;
    else
        heatGrade = camFor(models, s2.cnn.input, gradeTarget, mask);
    end
    camMs = round(1000 * toc(t0));

    counted = s2.rule.counts.MA + s2.rule.counts.HE + s2.rule.counts.EX + s2.rule.counts.SE;
    if isfield(models, 'policy'), policy = models.policy; else, policy = drscreen.reviewPolicy(); end
    agreement = attentionAgreement(heatRef, s1.masks, mask, counted, s2.fusion.referable, policy);

    overlays.gradcamReferable = overlayHeat(original, heatRef);
    overlays.gradcamGrade = overlayHeat(original, heatGrade);
    overlays.lesions = lesionOverlay(original, s1);
    overlays.vessels = labeloverlay(original, s1.masks.vessels, 'Colormap', [0 0.8 0], 'Transparency', 0.4);
    overlays.original = original;
    overlays.enhanced = s0.image;

    labels = drscreen.icdrLabels();
    s3 = struct('gradcamLayer', 'top_activation', ...
        'gradcamTargets', struct('referable', 'P(grade >= 2)', 'grade', sprintf('P(grade >= %d)', gradeTarget)), ...
        'attentionAgreement', agreement, ...
        'criteriaText', sprintf('Grade %d (%s) by ICDR criteria: %s', s2.rule.grade, labels{s2.rule.grade + 1}, s2.rule.criteriaText), ...
        'confidenceText', s2.fusion.confidenceText, 'overlays', overlays, 'heatReferable', heatRef, ...
        'elapsedMs', round(1000 * toc(t0)), 'gradcamMs', camMs);
end

function heat = camFor(models, x, outputIndex, mask)
    % Grad-CAM for ordinal head k. The grader's head is GAP -> Dense(sigmoid),
    % so d y_k / d A_c is spatially constant and equals y_k (1 - y_k) w_kc / HW:
    % Grad-CAM = ReLU(sum_c w_kc A_c) up to a positive scalar that the
    % normalisation removes. Exactly what stage3_explain._cam_function
    % computes with a gradient tape (verified: same maps on the samples).
    if isfield(models, 'graderFeatures') && ~isempty(models.graderFeatures) && ~isempty(models.graderHead)
        A = drscreen.predictNet(models.graderFeatures, x);            % 16 x 16 x 1536
        w = models.graderHead.dense_kernel(:, outputIndex);           % 1536 x 1
        map = max(sum(double(A) .* reshape(w, 1, 1, []), 3), 0);
        map = map / max(max(map(:)), 1e-8);
    else
        map = gradCAM(models.grader, x, @(y) y(outputIndex), 'FeatureLayer', 'top_activation');
    end
    heat = imresize(double(map), size(mask), 'bicubic');
    heat = min(max(heat, 0), 1);
    heat(~mask) = 0;
end

function out = overlayHeat(image, heat)
    cmap = jet(256);
    idx = uint8(round(min(max(heat, 0), 1) * 255)) + 1;
    colour = uint8(255 * reshape(cmap(idx, :), [size(heat) 3]));
    out = uint8(0.6 * double(image) + 0.4 * double(colour));
    keep = repmat(heat <= 0.02, [1 1 3]);
    out(keep) = image(keep);
end

function out = lesionOverlay(original, s1)
    out = original;
    colours = struct('MA', [255 0 0], 'HE', [140 0 0], 'EX', [255 220 0], 'SE', [255 255 255]);
    keys = fieldnames(colours);
    for i = 1:numel(keys)
        edge = bwperim(s1.masks.(keys{i}));
        edge = imdilate(edge, strel('disk', 1));
        for c = 1:3
            ch = out(:, :, c); ch(edge) = colours.(keys{i})(c); out(:, :, c) = ch;
        end
    end
    od = s1.opticDisc;
    out = insertShape(out, 'Circle', [od.centre, od.radius], 'Color', [0 200 255], 'LineWidth', 2);
    out = insertMarker(out, s1.fovea.centre, 'plus', 'Color', [0 200 255], 'Size', 9);
end

function a = attentionAgreement(heat, masks, fovMask, counted, referable, policy)
    union = masks.MA | masks.HE | masks.EX | masks.SE;
    lesionPixels = nnz(union);
    if lesionPixels == 0
        a = struct('score', NaN, 'lesionPixels', 0, 'flag', false, 'note', 'no lesions detected, so agreement is undefined');
        return;
    end
    wide = imdilate(union, strel('disk', 20));
    total = sum(heat(fovMask)); inside = sum(heat(wide));
    score = 0; if total > 1e-6, score = inside / total; end
    chance = nnz(wide & fovMask) / max(nnz(fovMask), 1);
    lift = 0; if chance > 0, lift = score / chance; end
    floorValue = policy.attentionFloor;
    low = score < floorValue && (isnan(policy.attentionMinLift) || lift < policy.attentionMinLift);
    applies = counted > 0 && referable;
    if low && applies
        note = 'attention not on lesions';
    elseif counted == 0
        note = 'only sub-threshold detections; not used for review';
    elseif ~referable
        note = 'not referable by CNN; heatmap not used for review';
    else
        note = 'attention overlaps lesion evidence';
    end
    a = struct('score', round(score * 1000) / 1000, 'chanceLevel', round(chance * 1000) / 1000, 'lift', round(lift * 100) / 100, ...
        'floor', floorValue, 'lesionPixels', lesionPixels, 'flag', low && applies, 'note', note);
end
