function meta = tierTable(code)
%TIERTABLE  Deadline, facility and queue rank per tier (architecture §9.2).
%   meta = drscreen.tierTable('P2') -> struct(label, deadlineDays, deadlineText,
%   facility, facilityTypes, rank). Queue order: P0, P1, P3, P2, P4.
    switch code
        case 'P0', meta = struct('label', 'Retake', 'deadlineDays', 0, 'deadlineText', 'same visit', ...
                'facility', 'PHC', 'facilityTypes', {{'phc'}}, 'rank', 0);
        case 'P1', meta = struct('label', 'Urgent', 'deadlineDays', 7, 'deadlineText', 'within 7 days', ...
                'facility', 'district hospital, in person', 'facilityTypes', {{'district_hospital'}}, 'rank', 1);
        case 'P3', meta = struct('label', 'Review', 'deadlineDays', 3, 'deadlineText', 'tele-review within 3 days, then re-tiered', ...
                'facility', 'tele-ophthalmology', 'facilityTypes', {{'tele', 'clinic', 'district_hospital'}}, 'rank', 2);
        case 'P2', meta = struct('label', 'Referable', 'deadlineDays', 30, 'deadlineText', 'within 30 days', ...
                'facility', 'nearest ophthalmology clinic or tele-review', 'facilityTypes', {{'clinic', 'district_hospital', 'tele'}}, 'rank', 3);
        case 'P4', meta = struct('label', 'Routine', 'deadlineDays', 365, 'deadlineText', '12-month recall', ...
                'facility', 'PHC camera', 'facilityTypes', {{'phc'}}, 'rank', 4);
        otherwise, error('drscreen:tier', 'unknown tier %s', code);
    end
end
