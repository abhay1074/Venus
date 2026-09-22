function s0 = gate(imageRGB, models)
%GATE  Stage 0: modality check, FOV normalisation, quality, enhancement.
%   s0 = drscreen.gate(imageRGB, models)
%   imageRGB : uint8 HxWx3 as read by imread
%   models   : struct from drscreen.loadModels (fields gate, quality may be [])
%
%   Fields of s0: accepted, stopReason, modality, quality, fov, image (512x512
%   working frame, enhanced when 'usable'), mask (512x512 logical), original
%   (512x512 before enhancement), elapsedMs. Mirrors backend/venus/stage0_gate.py.

    t0 = tic;
    modality = modalityCheck(imageRGB, models);
    if ~modality.accepted
        s0 = struct('accepted', false, 'stopReason', modality.reason, 'modality', modality, ...
            'quality', [], 'fov', [], 'elapsedMs', round(1000 * toc(t0)));
        return;
    end
    [image, mask, geometry] = drscreen.normaliseFov(imageRGB, 512);
    features = drscreen.qualityFeatures(image, mask, geometry);
    [label, score, reason, notes] = drscreen.qualityLabel(features);
    cnn = []; cnnWarning = '';
    if isfield(models, 'quality') && ~isempty(models.quality)
        probs = drscreen.predictNet(models.quality, imresize(image, [256 256], 'box'));         % cv2.INTER_AREA
        cnn = struct('good', double(probs(1)), 'usable', double(probs(2)), 'reject', double(probs(3)));
        if ~strcmp(label, 'reject')
            % As in stage0_gate.run: letting the EyeQ-trained CNN reject refused
            % 46 % of validation images (78 % of PDR), so it decides good vs
            % usable (whether to enhance) and its reject probability is a
            % warning; only the handcrafted hard limits reject.
            if cnn.good >= cnn.usable && cnn.reject < 0.5, label = 'good'; else, label = 'usable'; end
            if cnn.reject >= 0.5
                cnnWarning = sprintf('learned quality classifier: low quality (P = %.2f); enhanced and graded, read with care', cnn.reject);
                notes{end+1} = 'learned classifier: low quality';
            end
            score = round((cnn.good + 0.5 * cnn.usable) * 1000) / 1000;
        end
    end
    enhanced = strcmp(label, 'usable');
    if enhanced
        working = drscreen.enhance(image, mask);
    else
        working = image;
    end
    working(repmat(~mask, [1 1 3])) = 0;
    quality = struct('label', label, 'score', score, 'features', features, 'enhanced', enhanced, ...
        'notes', {notes}, 'retakeReason', reason, 'cnnProbabilities', cnn, 'cnnWarning', cnnWarning);
    s0 = struct('accepted', ~strcmp(label, 'reject'), 'stopReason', reason, 'modality', modality, ...
        'quality', quality, 'fov', geometry, 'image', working, 'mask', mask, 'original', image, ...
        'elapsedMs', round(1000 * toc(t0)));
end

function modality = modalityCheck(imageRGB, models)
    threshold = 0.987005;   % locked 2026-09-02, see stage0_gate.py
    classes = {'fundus', 'dermoscopy', 'face', 'sclera', 'other'};
    if ~isfield(models, 'gate') || isempty(models.gate)
        modality = struct('ran', false, 'accepted', false, 'fundusProbability', NaN, 'threshold', threshold, ...
            'reason', 'The modality gate could not be loaded, so the image cannot be verified as a fundus photograph.');
        return;
    end
    probs = drscreen.predictNet(models.gate, imresize(imageRGB, [224 224], 'bilinear', 'Antialiasing', false));   % cv2.resize default
    p = double(probs(1));
    accepted = p >= threshold;
    reason = '';
    if ~accepted, reason = 'This does not look like a fundus (retinal) photograph. Nothing was analysed.'; end
    modality = struct('ran', true, 'accepted', accepted, 'fundusProbability', p, ...
        'classProbabilities', cell2struct(num2cell(double(probs(:))), classes(:), 1), 'threshold', threshold, 'reason', reason);
end
