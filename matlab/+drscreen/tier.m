function t = tier(result, intake)
%TIER  Priority tier from the screening result and the intake risk factors.
%   t = drscreen.tier(result, intake)
%   result : screenImage result struct, or [] for a Stage 0 reject
%   intake : struct with optional fields symptoms (cellstr), pregnant, hba1c,
%            diabetesYears, hypertension, insulin
%
%   Tiers follow the ICDR follow-up guidance (architecture §9.2):
%     P0 retake     Stage 0 reject                          same visit, PHC
%     P1 urgent     PDR / P(NV) > 0.5 / sudden vision loss / grade 3 + symptoms   7 days
%     P3 review     human-review flag                        3 days tele-review
%     P2 referable  grade 2-3 or P(referable) >= threshold  30 days
%     P4 routine    grade 0-1                                12-month recall
%   Pregnancy and visual symptoms raise one tier (P4->P2, P2->P1); metabolic
%   factors shorten the routine recall to 6 months. Never lowered.

    if nargin < 2 || isempty(intake), intake = struct(); end
    symptoms = getfieldOr(intake, 'symptoms', {});
    reasons = {};

    if isempty(result) || ~result.accepted
        code = 'P0';
        reasons{end+1} = 'image rejected at Stage 0; retake on the same visit';
    else
        f = result.stage2.fusion; c = result.stage2.cnn; g = f.grade;
        if g == 4 || c.nvProbability > 0.5
            code = 'P1'; reasons{end+1} = 'PDR evidence (grade 4 or P(NV) > 0.5)';
        elseif any(strcmp(symptoms, 'sudden_vision_loss'))
            code = 'P1'; reasons{end+1} = 'sudden vision loss reported';
        elseif g == 3 && ~isempty(symptoms)
            code = 'P1'; reasons{end+1} = 'severe NPDR with symptoms';
        elseif f.flagForReview
            code = 'P3'; reasons{end+1} = ['human-review flag: ' strjoin(f.flagReasons, '; ')];
        elseif g >= 2 || f.referable
            code = 'P2'; reasons{end+1} = sprintf('grade %d / P(referable) %.2f >= threshold', g, f.pReferable);
        else
            code = 'P4'; reasons{end+1} = sprintf('grade %d, gradable, no symptoms', g);
        end
    end

    strong = {}; metabolic = {};
    if getfieldOr(intake, 'pregnant', false), strong{end+1} = 'pregnancy'; end
    if any(ismember(symptoms, {'blurred_vision', 'floaters'})), strong{end+1} = 'visual symptoms'; end
    if getfieldOr(intake, 'hba1c', 0) > 9, metabolic{end+1} = 'HbA1c > 9'; end
    if getfieldOr(intake, 'diabetesYears', 0) > 10, metabolic{end+1} = 'diabetes > 10 years'; end
    if getfieldOr(intake, 'hypertension', false), metabolic{end+1} = 'hypertension'; end
    if getfieldOr(intake, 'insulin', false), metabolic{end+1} = 'insulin use'; end

    recallMonths = 12;
    if ~isempty(strong) && any(strcmp(code, {'P4', 'P2'}))
        if strcmp(code, 'P4'), code = 'P2'; else, code = 'P1'; end
        reasons{end+1} = ['raised one tier by risk factors: ' strjoin(strong, ', ')];
    elseif strcmp(code, 'P4') && ~isempty(metabolic)
        recallMonths = 6;
        reasons{end+1} = ['recall shortened to 6 months: ' strjoin(metabolic, ', ')];
    end

    meta = drscreen.tierTable(code);
    risk = 0;
    if ~isempty(result) && result.accepted, risk = result.stage2.fusion.pReferable; end
    risk = min(risk + 0.05 * (numel(strong) + numel(metabolic)), 1);
    deadlineDays = meta.deadlineDays; deadlineText = meta.deadlineText;
    if strcmp(code, 'P4') && recallMonths == 6
        deadlineDays = 183; deadlineText = '6-month recall';
    end
    t = struct('tier', code, 'label', meta.label, 'deadlineDays', deadlineDays, ...
        'deadlineText', deadlineText, 'facility', meta.facility, 'riskScore', round(risk * 1000) / 1000, ...
        'riskFactors', {[strong metabolic]}, 'reasons', {reasons});
end

function v = getfieldOr(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end
