function modelName = buildDistrictModel(p, modelName, opts)
%BUILDDISTRICTMODEL  Build district_screening.slx (SimEvents) from a script.
%   modelName = buildDistrictModel(districtParams()) creates and saves the
%   SimEvents block diagram of the district screening programme:
%
%     Patient generator (Poisson) -> PHC queue -> Capture servers (cameras)
%       -> AI triage (Entity Output Switch on the `route` attribute)
%            port 1 -> Ophthalmologist queue (priority: AI-positive first)
%                      -> Doctor pool (DoctorPoolDES: K doctors with a daily
%                         hour budget) -> Result terminator
%            port 2 -> Outcome switch on `referable`
%                      port 1 -> Discharge terminator (12-month recall)
%                      port 2 -> Missed terminator (referable, never seen)
%
%   Each patient entity (bus type PatientBus) carries attributes drawn in the
%   generator's Generate action from the parameters: referable (true DR state
%   at the referable prevalence), aiPositive (measured sensitivity /
%   specificity), flagged (measured flag rate), toDoctor, readerCalls (human
%   reader sens/spec), service times, and route = 1 (doctor) / 2 (discharge)
%   because an Entity Output Switch reads the attribute as the port index.
%
%   The terminators' arrival counters are logged with To Workspace blocks
%   (missedCount, resultCount, dischargeCount), so a Simulink.SimulationOutput
%   carries them for runSweep. The parameter struct p is written to the model
%   workspace; the DoctorPoolDES block parameters are expressions on it, so
%   the sweep (parsim + Simulink.SimulationInput) only changes `params`.
%
%   Verified in R2026a: library `sldelib`, block names with a newline
%   ('Entity' newline 'Output Switch'), parameters TimeSource = 'MATLAB
%   action' + IntergenerationTimeAction, EntityType = 'Bus object' +
%   EntityTypeName, NumberOutputPorts, SwitchAttributeName. Every set_param is
%   still wrapped so a future rename prints one line instead of aborting.
%   Pass struct('plainServer', true) as a third argument to build the older
%   Entity Server variant (no daily budget) for comparison.

    if nargin < 1, p = districtParams(); end
    if nargin < 2, modelName = 'district_screening'; end
    if nargin < 3, opts = struct(); end
    plainServer = isfield(opts, 'plainServer') && opts.plainServer;
    here = fileparts(mfilename('fullpath'));
    addpath(here);                                    % DoctorPoolDES.m must be on the path
    assignin('base', 'PatientBus', districtBus());    % entity type (bus objects live in the base workspace)

    if bdIsLoaded(modelName), close_system(modelName, 0); end
    new_system(modelName); open_system(modelName);
    lib = 'sldelib';                                  % the SimEvents library (`simevents` opens it)
    add = @(blk, name, pos) add_block([lib '/' blk], [modelName '/' name], 'Position', pos);
    sink = @(name, pos) add_block('simulink/Sinks/To Workspace', [modelName '/' name], 'Position', pos);

    gen   = add('Entity Generator',      'Patient generator',      [40 100 120 140]);
    phcQ  = add('Entity Queue',          'PHC queue',              [180 100 260 140]);
    cam   = add('Entity Server',         'Capture + Stage 0',      [320 100 400 140]);
    ai    = add(['Entity' newline 'Output Switch'], 'AI triage', [460 100 540 140]);
    docQ  = add('Entity Queue',          'Ophthalmologist queue',  [620 40 700 80]);
    if plainServer
        doc  = add('Entity Server',      'Tele-review + in-person', [760 40 840 80]);
    else
        doc  = add(['MATLAB' newline 'Discrete-Event System'], 'Doctor pool (daily budget)', [760 40 880 80]);
    end
    res   = add('Entity Terminator',     'Result',                 [940 40 1000 80]);
    outc  = add(['Entity' newline 'Output Switch'], 'Outcome',    [620 160 700 200]);
    dis   = add('Entity Terminator',     'Discharge + recall',     [780 140 840 180]);
    mis   = add('Entity Terminator',     'Missed',                 [780 200 840 240]);
    wRes  = sink('resultCount',    [1060 40 1130 80]);
    wDis  = sink('dischargeCount', [900 140 970 180]);
    wMis  = sink('missedCount',    [900 200 970 240]);

    dayMinutes = p.phcHoursPerDay * 60;
    meanInterarrival = (p.workingDays * dayMinutes) / p.annualPatients;

    trySet(gen, 'GenerationMethod', 'Time-based');
    trySet(gen, 'TimeSource', 'MATLAB action');
    trySet(gen, 'IntergenerationTimeAction', sprintf('dt = -%g * log(rand);', meanInterarrival));   % exponential
    trySet(gen, 'EntityType', 'Bus object');
    trySet(gen, 'EntityTypeName', 'PatientBus');
    trySet(gen, 'GenerateAction', generateAction(p));

    trySet(phcQ, 'Capacity', 'inf');
    trySet(phcQ, 'QueueType', 'FIFO');
    trySet(cam, 'Capacity', num2str(p.phcs * min(p.camerasPerPhc, p.operatorsPerPhc)));
    trySet(cam, 'ServiceTimeSource', 'Attribute');
    trySet(cam, 'ServiceTimeAttributeName', 'captureMinutes');

    trySet(ai, 'NumberOutputPorts', '2');
    trySet(ai, 'SwitchingCriterion', 'From attribute');
    trySet(ai, 'SwitchAttributeName', 'route');       % 1 -> doctor, 2 -> discharge (attribute value = port index)
    trySet(outc, 'NumberOutputPorts', '2');
    trySet(outc, 'SwitchingCriterion', 'From attribute');
    trySet(outc, 'SwitchAttributeName', 'outcome');   % 1 -> discharge, 2 -> missed (referable, never seen)

    trySet(docQ, 'Capacity', 'inf');
    trySet(docQ, 'QueueType', 'Priority');
    trySet(docQ, 'PrioritySource', 'aiPositive');
    trySet(docQ, 'SortingDirection', 'Descending');
    if plainServer
        trySet(doc, 'Capacity', num2str(p.ophthalmologists));
        trySet(doc, 'ServiceTimeSource', 'Attribute');
        trySet(doc, 'ServiceTimeAttributeName', 'reviewMinutes');
    else
        % The System object's public properties are the block parameters;
        % expressions refer to the model workspace so parsim only sets `params`.
        trySet(doc, 'System', 'DoctorPoolDES');
        trySet(doc, 'NumDoctors', 'params.ophthalmologists');
        trySet(doc, 'DailyBudgetMinutes', 'params.doctorHoursPerDay * 60');
        trySet(doc, 'DayMinutes', 'params.phcHoursPerDay * 60');
    end

    for blk = {res, dis, mis}
        trySet(blk{1}, 'NumberEntitiesArrived', 'on');     % statistic output port
    end
    for blk = {wRes, wDis, wMis}
        trySet(blk{1}, 'VariableName', get_param(blk{1}, 'Name'));
        trySet(blk{1}, 'SaveFormat', 'Timeseries');
    end

    connect = @(a, b, pa, pb) add_line(modelName, sprintf('%s/%d', get_param(a, 'Name'), pa), sprintf('%s/%d', get_param(b, 'Name'), pb), 'autorouting', 'on');
    connect(gen, phcQ, 1, 1); connect(phcQ, cam, 1, 1); connect(cam, ai, 1, 1);
    connect(ai, outc, 2, 1); connect(outc, dis, 1, 1); connect(outc, mis, 2, 1); connect(doc, res, 1, 1);
    connect(ai, docQ, 1, 1); connect(docQ, doc, 1, 1);
    % Statistic ports are signal ports after the entity port.
    connect(res, wRes, 1, 1); connect(dis, wDis, 1, 1); connect(mis, wMis, 1, 1);

    mws = get_param(modelName, 'ModelWorkspace');
    assignin(mws, 'params', p);
    % The entity bus must exist in the base workspace whenever the model is
    % loaded or simulated (parsim workers included): recreate it from districtBus.m.
    set_param(modelName, 'PreLoadFcn', 'assignin(''base'', ''PatientBus'', districtBus());', ...
                         'InitFcn', 'assignin(''base'', ''PatientBus'', districtBus());');
    set_param(modelName, 'StopTime', num2str(p.workingDays * dayMinutes), 'SolverType', 'Variable-step', 'Solver', 'VariableStepDiscrete');
    save_system(modelName, fullfile(here, [modelName '.slx']));
    fprintf('saved %s.slx\n', modelName);
end

function trySet(block, name, value)
    try
        set_param(block, name, value);
    catch err
        fprintf('  set_param(%s, ''%s'') failed: %s\n', strrep(get_param(block, 'Name'), newline, ' '), name, err.message);
    end
end
