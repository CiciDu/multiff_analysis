function bytes = deep_size(x)
    info = whos('x');
    bytes = info.bytes;

    if isstruct(x)
        fn = fieldnames(x);
        for i = 1:numel(fn)
            for k = 1:numel(x)
                bytes = bytes + deep_size(x(k).(fn{i}));
            end
        end
    elseif iscell(x)
        for i = 1:numel(x)
            bytes = bytes + deep_size(x{i});
        end
    end
end
