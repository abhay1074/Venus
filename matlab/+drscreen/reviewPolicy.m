function policy = reviewPolicy(configDir)
%REVIEWPOLICY  Read config/review_policy.json (validation-chosen review rules).
%   policy = drscreen.reviewPolicy() mirrors backend.venus.config.review_policy:
%   the abstain band half-width in logit space and the attention floor chosen
%   by backend/eval/review_policy.py on the validation sample. When the file
%   is absent the architecture's defaults apply (+/-0.05 probability band from
%   the operating point, attention floor 0.15 with the lift condition).
%
%   Fields: abstainBandLogit (NaN when absent), attentionFloor,
%   attentionMinLift (NaN when the policy file exists), chosenOn, raw.

    if nargin < 1
        configDir = fullfile(drscreen.repoRoot(), 'backend', 'config');
    end
    path = fullfile(configDir, 'review_policy.json');
    policy = struct('abstainBandLogit', NaN, 'attentionFloor', 0.15, 'attentionMinLift', 1.5, ...
        'chosenOn', '', 'raw', struct());
    if ~isfile(path)
        return;
    end
    raw = jsondecode(fileread(path));
    policy.raw = raw;
    if isfield(raw, 'abstain_band_logit') && ~isempty(raw.abstain_band_logit)
        policy.abstainBandLogit = raw.abstain_band_logit;
    end
    if isfield(raw, 'attention_floor') && ~isempty(raw.attention_floor)
        policy.attentionFloor = raw.attention_floor;
    end
    policy.attentionMinLift = NaN;   % the floor alone was chosen on validation
    if isfield(raw, 'chosen_on'), policy.chosenOn = raw.chosen_on; end
end
