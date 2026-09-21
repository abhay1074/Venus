function bus = districtBus()
%DISTRICTBUS  The patient entity type of district_screening.slx: one double per attribute.
%   Assigned to the base workspace as PatientBus by the model's PreLoadFcn /
%   InitFcn (buildDistrictModel sets them), so loading or simulating the model
%   anywhere - a parsim worker included - recreates it.
    names = {'referable', 'aiPositive', 'flagged', 'toDoctor', 'route', 'readerCalls', 'captureMinutes', 'reviewMinutes', 'outcome'};
    elems = Simulink.BusElement.empty(0, 1);
    for i = 1:numel(names)
        e = Simulink.BusElement; e.Name = names{i}; e.DataType = 'double'; e.Dimensions = 1;
        elems(i, 1) = e;
    end
    bus = Simulink.Bus; bus.Elements = elems;
end
