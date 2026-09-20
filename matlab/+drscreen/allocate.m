function [decisions, slots] = allocate(queue, slots, now, facilities)
%ALLOCATE  Book the earliest feasible slot for every queued patient.
%   [decisions, slots] = drscreen.allocate(queue, slots, now, facilities)
%
%   queue      : struct array with fields id, tier ('P1'...), riskScore,
%                queuedAt (datetime), deadlineDays, villagePos [x y]
%   slots      : struct array with fields id, facilityId, startsAt (datetime),
%                booked (logical), appointmentId, holder (struct or [] with
%                fields id, tier, queuedAt, deadlineDays) -- mutated and returned
%   now        : datetime
%   facilities : containers.Map facilityId -> struct(type, pos [x y])
%
%   Queue key: (overdue first, tier rank, -risk, -waiting time), so ageing lets
%   a P3 past its deadline outrank a fresh P2 and no tier is starved. Each item
%   takes the earliest open slot at the nearest facility able to serve its
%   tier, inside its deadline. If none exists, the lowest-priority booked
%   holder whose own deadline still allows re-slotting is bumped to a later
%   slot (and the bump is recorded). An overdue item's window is re-opened
%   from now so it is booked, not waitlisted. Pure function: no I/O.

    order = {'P0', 'P1', 'P3', 'P2', 'P4'};
    rank = @(t) find(strcmp(order, t)) - 1;
    n = numel(queue);
    keys = zeros(n, 4);
    for i = 1:n
        waitingH = hours(now - queue(i).queuedAt);
        deadlineH = queue(i).deadlineDays * 24;
        keys(i, :) = [~(waitingH > deadlineH), rank(queue(i).tier), -queue(i).riskScore, -waitingH];
    end
    [~, idx] = sortrows(keys);
    decisions = struct('appointmentId', {}, 'slotId', {}, 'facilityId', {}, 'startsAt', {}, 'bumped', {});

    for k = idx'
        item = queue(k);
        window = days(item.deadlineDays);
        deadline = item.queuedAt + window;
        if deadline <= now
            deadline = now + max(window, days(1));
        end
        meta = drscreen.tierTable(item.tier); allowed = meta.facilityTypes;
        best = 0; bestKey = [];
        for s = 1:numel(slots)
            sl = slots(s);
            if sl.booked || sl.startsAt <= now || sl.startsAt > deadline, continue; end
            fac = facilities(sl.facilityId);
            if ~any(strcmp(allowed, fac.type)), continue; end
            key = [floor(datenum(sl.startsAt)), norm(item.villagePos - fac.pos), datenum(sl.startsAt)];
            if best == 0 || isLess(key, bestKey), best = s; bestKey = key; end
        end
        if best > 0
            slots(best).booked = true; slots(best).appointmentId = item.id;
            slots(best).holder = struct('id', item.id, 'tier', item.tier, 'queuedAt', item.queuedAt, 'deadlineDays', item.deadlineDays);
            decisions(end+1) = struct('appointmentId', item.id, 'slotId', slots(best).id, ...
                'facilityId', slots(best).facilityId, 'startsAt', slots(best).startsAt, 'bumped', []); %#ok<AGROW>
            continue;
        end
        % No slot inside the deadline: bump the lowest-priority holder whose
        % own deadline still allows a later slot.
        candidates = [];
        for s = 1:numel(slots)
            sl = slots(s);
            if ~sl.booked || isempty(sl.holder) || sl.startsAt <= now || sl.startsAt > deadline, continue; end
            if ~any(strcmp(allowed, facilities(sl.facilityId).type)), continue; end
            if rank(sl.holder.tier) <= rank(item.tier), continue; end
            candidates(end+1, :) = [rank(sl.holder.tier), s]; %#ok<AGROW>
        end
        bumped = [];
        if ~isempty(candidates)
            candidates = sortrows(candidates, -1);
            for c = 1:size(candidates, 1)
                s = candidates(c, 2); holder = slots(s).holder;
                holderDeadline = holder.queuedAt + days(holder.deadlineDays);
                holderMeta = drscreen.tierTable(holder.tier); holderTypes = holderMeta.facilityTypes;
                spare = 0; spareT = [];
                for t = 1:numel(slots)
                    st = slots(t);
                    if st.booked || st.startsAt <= slots(s).startsAt || st.startsAt > holderDeadline, continue; end
                    if ~any(strcmp(holderTypes, facilities(st.facilityId).type)), continue; end
                    if spare == 0 || st.startsAt < spareT, spare = t; spareT = st.startsAt; end
                end
                if spare > 0
                    slots(spare).booked = true; slots(spare).appointmentId = holder.id; slots(spare).holder = holder;
                    slots(s).appointmentId = item.id;
                    slots(s).holder = struct('id', item.id, 'tier', item.tier, 'queuedAt', item.queuedAt, 'deadlineDays', item.deadlineDays);
                    bumped = struct('appointmentId', holder.id, 'fromSlot', slots(s).id, 'toSlot', slots(spare).id, 'startsAt', spareT);
                    decisions(end+1) = struct('appointmentId', item.id, 'slotId', slots(s).id, ...
                        'facilityId', slots(s).facilityId, 'startsAt', slots(s).startsAt, 'bumped', bumped); %#ok<AGROW>
                    break;
                end
            end
        end
        if isempty(bumped)
            decisions(end+1) = struct('appointmentId', item.id, 'slotId', '', 'facilityId', '', 'startsAt', NaT, 'bumped', []); %#ok<AGROW>
        end
    end
end

function tf = isLess(a, b)
    tf = false;
    for i = 1:numel(a)
        if a(i) < b(i), tf = true; return; elseif a(i) > b(i), return; end
    end
end
