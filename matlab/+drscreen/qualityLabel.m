function [label, score, reason, notes] = qualityLabel(f)
%QUALITYLABEL  good / usable / reject from the handcrafted features.
%   [label, score, reason, notes] = drscreen.qualityLabel(features)
%   Fixed limits (QUALITY_LIMITS in stage0_gate.py); `reason` is the
%   operator-facing retake sentence when label is 'reject'.

    L = struct('sharpnessReject', 6.0, 'sharpnessUsable', 14.0, 'illuminationReject', 0.28, ...
        'illuminationUsable', 0.45, 'saturatedReject', 0.12, 'saturatedUsable', 0.05, ...
        'darkReject', 0.55, 'darkUsable', 0.35, 'coverageReject', 0.30, 'circularityReject', 0.55);
    rejects = {}; notes = {};
    if f.sharpness < L.sharpnessReject, rejects{end+1} = 'Image too blurry - hold still and refocus';
    elseif f.sharpness < L.sharpnessUsable, notes{end+1} = 'slightly soft focus'; end
    if f.illuminationUniformity < L.illuminationReject, rejects{end+1} = 'Uneven illumination - centre the pupil and check the flash';
    elseif f.illuminationUniformity < L.illuminationUsable, notes{end+1} = 'vignetting'; end
    if f.saturatedFraction > L.saturatedReject, rejects{end+1} = 'Overexposed - reduce flash intensity';
    elseif f.saturatedFraction > L.saturatedUsable, notes{end+1} = 'bright reflections'; end
    if f.darkFraction > L.darkReject, rejects{end+1} = 'Underexposed - increase illumination or dilate';
    elseif f.darkFraction > L.darkUsable, notes{end+1} = 'dark exposure'; end
    if f.fovCoverage < L.coverageReject, rejects{end+1} = 'Retina fills too little of the frame - move closer or zoom'; end
    if f.fovCircularity < L.circularityReject, rejects{end+1} = 'Partial capture - the field of view is cut off; re-centre'; end

    clip = @(v) min(max(v, 0), 1);
    score = mean([clip(f.sharpness / 40), clip((f.illuminationUniformity - 0.2) / 0.6), ...
        clip(1 - f.saturatedFraction / 0.15), clip(1 - f.darkFraction / 0.6), clip((f.fovCircularity - 0.4) / 0.5)]);
    score = round(score * 1000) / 1000;
    if ~isempty(rejects)
        label = 'reject'; reason = rejects{1};
    elseif ~isempty(notes)
        label = 'usable'; reason = '';
    else
        label = 'good'; reason = '';
    end
end
