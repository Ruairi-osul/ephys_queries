SELECT waveforms.neuron_id, waveforms.waveform_index, waveforms.waveform_value
FROM waveforms 
JOIN neurons 
    ON neurons.id = waveforms.neuron_id
JOIN recording_sessions
    ON neurons.recording_session_id = recording_sessions.id
JOIN experimental_groups
    ON recording_sessions.group_id = experimental_groups.id
WHERE experimental_groups.experiment_id=2;