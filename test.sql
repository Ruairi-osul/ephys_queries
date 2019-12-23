SELECT 
    -- discrete_signal_data.signal_id, discrete_signal_data.timepoint_sample
    -- recording_sessions.session_name, COUNT(*)
    discrete_signals.signal_name, experiments.experiment_name, experimental_groups.group_name
FROM discrete_signal_data
    JOIN session_discrete_signals
        ON session_discrete_signals.id=discrete_signal_data.signal_id
    JOIN discrete_signals
        ON discrete_signals.id=session_discrete_signals.discrete_signal_id
    JOIN recording_sessions
        ON recording_sessions.id=session_discrete_signals.recording_session_id
    INNER JOIN recording_session_block_times
        ON recording_session_block_times.recording_session_id=recording_sessions.id
    JOIN experimental_groups
        ON experimental_groups.id=recording_sessions.group_id
    JOIN experiments 
        ON experiments.id=experimental_groups.experiment_id;
-- GROUP BY recording_sessions.session_name;