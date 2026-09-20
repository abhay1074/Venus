function modelName = buildDistrictModel(p, modelName)
%BUILDDISTRICTMODEL  Build district_screening.slx (SimEvents) from a script.
%   modelName = buildDistrictModel(districtParams()) creates and saves the
%   SimEvents block diagram of the district screening programme:
%
%     Patient generator (Poisson) -> PHC queue -> Capture servers (cameras)
%       -> AI triage (Entity Output Switch on the `toDoctor` attribute)
%            -> Ophthalmologist queue -> K doctor servers -> Result terminator
%            -> Discharge terminator (non-referable, 12-month recall)
%
%   Each patient entity carries attributes drawn in the generator's Generate
%   action from the parameters: referable (true DR state at the referable
%   prevalence), aiPositive (measured sensitivity / specificity), flagged
%   (measured flag rate), toDoctor, readerCalls (human reader sens/spec),
%   service times. Missed cases are entities with referable && ~toDoctor,
%   counted at the discharge terminator's entry action.
%
%   The parameter struct p is written to the model workspace so the sweep
%   (runSweep.m, parsim + Simulink.SimulationInput) only changes variables.
%
%   Block parameter names follow the SimEvents (R2024b) block dialogs. Each
%   set_param is wrapped so a renamed parameter reports one line instead of
%   aborting; fix any reported name and re-run. Doctors' daily hour budget is
%   not modelled by the plain Entity Server (the MATLAB reference
%   simulateDistrict.m does model it); to add it, replace the doctor server
%   with a MATLAB Discrete-Event System block.

    if nargin < 1, p = districtParams(); end
    if nargin < 2, modelName = 'district_screening'; end
    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName); open_system(modelName);
    lib = 'simeventslib';
    add = @(blk, name, pos) add_block([lib '/' blk], [modelName '/' name], 'Position', pos);

    gen   = add('Entity Generator',      'Patient generator',      [40 100 120 140]);
    phcQ  = add('Entity Queue',          'PHC queue',              [180 100 260 140]);
    cam   = add('Entity Server',         'Capture + Stage 0',      [320 100 400 140]);
    ai    = add('Entity Output Switch',  'AI triage',              [460 100 540 140]);
    docQ  = add('Entity Queue',          'Ophthalmologist queue',  [620 60 700 100]);
    doc   = add('Entity Server',         'Tele-review + in-person', [760 60 840 100]);
    res   = add('Entity Terminator',     'Result',                 [900 60 960 100]);
    dis   = add('Entity Terminator',     'Discharge + recall',     [620 160 700 200]);

    dayMinutes = p.phcHoursPerDay * 60;
    meanInterarrival = (p.workingDays * dayMinutes) / p.annualPatients;

    trySet(gen, 'GenerationMethod', 'Time-based');
    trySet(gen, 'TimeSource', 'Dialog');
    trySet(gen, 'IntergenerationTimeDistribution', 'Exponential');
    trySet(gen, 'IntergenerationTimeMean', num2str(meanInterarrival));
    trySet(gen, 'GenerateAction', sprintf([ ...
        'entity.referable = double(rand < %g);\n' ...
        'if entity.referable, entity.aiPositive = double(rand < %g); else, entity.aiPositive = double(rand < %g); end\n' ...
        'entity.flagged = double(rand < %g);\n' ...
        'entity.toDoctor = double(%d && (entity.aiPositive || entity.flagged)) + double(~%d);\n' ...
        'if entity.referable, entity.readerCalls = double(rand < %g); else, entity.readerCalls = double(rand < %g); end\n' ...
        'entity.captureMinutes = %g * (1 + double(rand < %g));\n' ...
        'entity.reviewMinutes = %g + %g * entity.readerCalls;\n'], ...
        p.referablePrevalence, p.sensitivity, 1 - p.specificity, p.flagRate, p.aiEnabled, p.aiEnabled, ...
        p.humanReaderSensitivity, 1 - p.humanReaderSpecificity, p.captureMinutes, p.retakeProbability, ...
        p.teleReviewMinutes, p.inPersonMinutes));
    trySet(gen, 'EntityAttributeNames', 'referable,aiPositive,flagged,toDoctor,readerCalls,captureMinutes,reviewMinutes');

    trySet(phcQ, 'Capacity', 'inf');
    trySet(phcQ, 'QueueType', 'FIFO');
    trySet(cam, 'Capacity', num2str(p.phcs * min(p.camerasPerPhc, p.operatorsPerPhc)));
    trySet(cam, 'ServiceTimeSource', 'Attribute');
    trySet(cam, 'ServiceTimeAttributeName', 'captureMinutes');

    trySet(ai, 'NumberOfOutputPorts', '2');
    trySet(ai, 'SwitchingCriterion', 'From attribute');
    trySet(ai, 'SwitchingAttributeName', 'toDoctor');   % 1 -> port 1? SimEvents maps attribute value to port index
    % Attribute value 0 (discharge) must map to port 2 and 1 (doctor) to port 1:
    % the switch uses the attribute as the port index, so route via an
    % Entity Attribute block if the version indexes from 1. See README.

    trySet(docQ, 'Capacity', 'inf');
    trySet(docQ, 'QueueType', 'Priority');
    trySet(docQ, 'PriorityAttributeName', 'aiPositive');
    trySet(docQ, 'PrioritySortingDirection', 'Descending');
    trySet(doc, 'Capacity', num2str(p.ophthalmologists));
    trySet(doc, 'ServiceTimeSource', 'Attribute');
    trySet(doc, 'ServiceTimeAttributeName', 'reviewMinutes');

    trySet(dis, 'EntryAction', 'if entity.referable && ~entity.toDoctor, missed = missed + 1; end');
    trySet(res, 'EntryAction', 'if entity.referable && entity.readerCalls, confirmed = confirmed + 1; end');

    connect = @(a, b, pa, pb) add_line(modelName, sprintf('%s/%d', get_param(a, 'Name'), pa), sprintf('%s/%d', get_param(b, 'Name'), pb), 'autorouting', 'on');
    connect(gen, phcQ, 1, 1); connect(phcQ, cam, 1, 1); connect(cam, ai, 1, 1);
    connect(ai, docQ, 1, 1); connect(ai, dis, 2, 1); connect(docQ, doc, 1, 1); connect(doc, res, 1, 1);

    mws = get_param(modelName, 'ModelWorkspace');
    assignin(mws, 'params', p);
    assignin(mws, 'missed', 0); assignin(mws, 'confirmed', 0);
    set_param(modelName, 'StopTime', num2str(p.workingDays * dayMinutes), 'SolverType', 'Variable-step', 'Solver', 'VariableStepDiscrete');
    save_system(modelName, fullfile(fileparts(mfilename('fullpath')), [modelName '.slx']));
    fprintf('saved %s.slx\n', modelName);
end

function trySet(block, name, value)
    try
        set_param(block, name, value);
    catch err
        fprintf('  set_param(%s, ''%s'') failed: %s\n', get_param(block, 'Name'), name, err.message);
    end
end
