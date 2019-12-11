import pandas as pd
import numpy as np
from sqlalchemy.sql import select, and_, or_


def select_recording_sessions(
    engine,
    metadata,
    group_names=None,
    exp_names=None,
    session_names=None,
    exclude_excluded_recordings=True,
    as_df=True,
):
    r_sesh, groups, experiments = (
        metadata.tables["recording_sessions"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    stmt = select([r_sesh, groups.c.group_name, experiments.c.experiment_name])
    stmt = stmt.select_from(r_sesh.join(groups).join(experiments))
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))
    if session_names:
        stmt = stmt.where(r_sesh.c.session_name.in_(session_names))

    with engine.connect() as con:
        rp = con.execute(stmt)

    if as_df:
        return _result_proxy_to_df(rp)
    else:
        return rp


def select_neurons(
    engine,
    metadata,
    group_names=None,
    exp_names=None,
    session_names=None,
    exclude_mua=True,
    as_df=True,
    exclude_excluded_recordings=True,
):
    neurons, r_sesh, groups, experiments = (
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    stmt = select(
        [
            neurons,
            r_sesh.c.session_name,
            groups.c.group_name,
            experiments.c.experiment_name,
        ]
    )
    stmt = stmt.select_from(neurons.join(r_sesh).join(groups).join(experiments))
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if exclude_mua:
        stmt = stmt.where(neurons.c.is_single_unit == 1)
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))
    if session_names:
        stmt = stmt.where(r_sesh.c.exp_names.in_(session_names))

    with engine.connect() as con:
        rp = con.execute(stmt)

    if as_df:
        return _result_proxy_to_df(rp)
    else:
        return rp


def select_ifr(
    engine,
    metadata,
    block_name="all",
    t_before=0,
    t_after=0,
    neuron_ids=None,
    group_names=None,
    exp_names=None,
    session_names=None,
    exclude_mua=True,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):

    ifr, neurons, r_sesh, rs_blocks, groups, experiments = (
        metadata.tables["neuron_ifr"],
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    stmt_block = select(
        [
            rs_blocks.c.recording_session_id,
            rs_blocks.c.block_start_samples,
            rs_blocks.c.block_end_samples,
        ]
    )
    stmt_block = stmt_block.select_from(
        rs_blocks.join(r_sesh).join(groups).join(experiments)
    )
    if block_name != "all":
        stmt_block = stmt_block.where(rs_blocks.c.block_name == block_name)
    stmt_block = stmt_block.alias("block")

    columns: list = [ifr.c.neuron_id, ifr.c.ifr]
    if align_to_block:
        columns.append(
            (ifr.c.timepoint_s - (stmt_block.c.block_start_samples / 30000)).label(
                "timepoint_s"
            )
        )
    else:
        columns.append(ifr.c.timepoint_s)

    if not exclude_mua:
        columns.extend([neurons.c.is_single_unit])
    if session_names:
        columns.extend([r_sesh.c.session_name])
    if group_names:
        columns.extend([groups.c.group_name])
    if exp_names:
        columns.extend([experiments.c.experiment_name, groups.c.group_name])

    stmt = select(columns).select_from(
        ifr.join(neurons)
        .join(r_sesh)
        .join(rs_blocks)
        .join(groups)
        .join(experiments)
        .join(stmt_block, stmt_block.c.recording_session_id == r_sesh.c.id)
    )
    if block_name != "all":
        stmt = stmt.where(
            and_(
                ifr.c.timepoint_s
                > (stmt_block.c.block_start_samples / 30000) - t_before,
                ifr.c.timepoint_s < (stmt_block.c.block_end_samples / 30000) + t_after,
            )
        )
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if neuron_ids:
        stmt = stmt.where(neurons.c.id.in_(neuron_ids))
    if session_names:
        stmt = stmt.where(r_sesh.c.session_name.in_(session_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment.in_(exp_names))

    with engine.connect() as conn:
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
        res.ifr, res.timepoint_s = (
            res.ifr.astype(np.float),
            res.timepoint_s.astype(np.float),
        )
    return res


def select_spike_times(
    engine,
    metadata,
    block_name="all",
    t_before=0,
    t_after=0,
    neuron_ids=None,
    group_names=None,
    exp_names=None,
    session_names=None,
    exclude_mua=True,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):
    spike_times, neurons, r_sesh, rs_blocks, groups, experiments = (
        metadata.tables["spike_times"],
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    stmt_block = select(
        [
            rs_blocks.c.recording_session_id,
            rs_blocks.c.block_start_samples,
            rs_blocks.c.block_end_samples,
        ]
    )
    stmt_block = stmt_block.select_from(
        rs_blocks.join(r_sesh).join(groups).join(experiments)
    )
    if block_name != "all":
        stmt_block = stmt_block.where(rs_blocks.c.block_name == block_name)
    stmt_block = stmt_block.alias("block")

    columns: list = [spike_times.c.neuron_id]

    if align_to_block:
        columns.append(
            (spike_times.c.spike_time_samples - stmt_block.c.block_start_samples).label(
                "spike_time_samples"
            )
        )
    else:
        columns.append(spike_times.c.spike_time_samples)

    if not exclude_mua:
        columns.extend([neurons.c.is_single_unit])
    if session_names:
        columns.extend([r_sesh.c.session_name])
    if group_names:
        columns.extend([groups.c.group_name])
    if exp_names:
        columns.extend([experiments.c.experiment_name, groups.c.group_name])

    stmt = select(columns).select_from(
        spike_times.join(neurons)
        .join(r_sesh)
        .join(rs_blocks)
        .join(groups)
        .join(experiments)
        .join(stmt_block, stmt_block.c.recording_session_id == r_sesh.c.id)
    )

    if block_name != "all":
        stmt = stmt.where(
            and_(
                spike_times.c.spike_time_samples
                > (stmt_block.c.block_start_samples - (t_before * 30000)),
                spike_times.c.spike_time_samples
                < stmt_block.c.block_end_samples + (t_after * 30000),
            )
        )
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if neuron_ids:
        stmt = stmt.where(neurons.c.id.in_(neuron_ids))
    if session_names:
        stmt = stmt.where(r_sesh.c.session_name.in_(session_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))

    with engine.connect() as conn:
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
        res.spike_time_samples = res.spike_time_samples.astype(np.int64)
    return res


def select_waveforms(
    engine,
    metadata,
    units="samples",
    group_names=None,
    neuron_ids=None,
    exp_names=None,
    session_names=None,
    exclude_mua=True,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):
    waveforms, neurons, r_sesh, groups, experiments = (
        metadata.tables["waveforms"],
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    columns = [waveforms.c.neuron_id, waveforms.c.waveform_value]
    if units == "seconds":
        columns.append((waveforms.c.waveform_index / 30000).label("waveform_index"))
    elif units == "samples":
        columns.append(waveforms.c.waveform_index)
    else:
        raise ValueError("Must Enter units of seconds or samples")

    if not exclude_mua:
        columns.extend([neurons.c.is_single_unit])
    if session_names:
        columns.extend([r_sesh.c.session_name])
    if group_names:
        columns.extend([groups.c.group_name])
    if exp_names:
        columns.extend([experiments.c.experiment_name, groups.c.group_name])

    stmt = select(columns)
    stmt = stmt.select_from(
        waveforms.join(neurons).join(r_sesh).join(groups).join(experiments)
    )

    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if neuron_ids:
        stmt = stmt.where(neurons.c.id.in_(neuron_ids))
    if session_names:
        stmt = stmt.where(r_sesh.c.session_name.in_(session_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))

    with engine.connect() as conn:
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
        res.waveform_value, res.waveform_index = (
            res.waveform_value.astype(np.float),
            res.waveform_index.astype(np.float),
        )
    return res


def select_analog_signal_data(
    engine,
    metadata,
    signal_names=None,
    block_name="pre",
    t_before=0,
    t_after=0,
    group_names=None,
    exp_names=None,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):

    a_data, sesh_a_sig, a_sigs, r_sesh, rs_blocks, groups, experiments = (
        metadata.tables["analog_data"],
        metadata.tables["session_analog_signals"],
        metadata.tables["analog_signals"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    stmt_block = select(
        [
            rs_blocks.c.recording_session_id,
            rs_blocks.c.block_start_samples,
            rs_blocks.c.block_end_samples,
        ]
    )
    stmt_block = stmt_block.select_from(
        rs_blocks.join(r_sesh).join(groups).join(experiments)
    )
    if block_name != "all":
        stmt_block = stmt_block.where(rs_blocks.c.block_name == block_name)
    stmt_block = stmt_block.alias("block")

    columns = [
        a_data.c.voltage,
        r_sesh.c.session_name,
        a_sigs.c.signal_name,
    ]
    if align_to_block:
        columns.append(
            (a_data.c.timepoint_s - (stmt_block.c.block_start_samples / 30000)).label(
                "timepoint_s"
            )
        )
    else:
        columns.append(a_data.c.timepoint_s)
    if group_names:
        columns.extend([groups.c.group_name])
    if exp_names:
        columns.extend([experiments.c.experiment_name, groups.c.group_name])

    stmt = select(columns)
    stmt = stmt.select_from(
        a_data.join(sesh_a_sig)
        .join(a_sigs)
        .join(r_sesh, r_sesh.c.id == sesh_a_sig.c.recording_session_id)
        .join(rs_blocks)
        .join(groups)
        .join(experiments)
        .join(
            stmt_block, stmt_block.c.recording_session_id == r_sesh.c.id, isouter=False
        )
    )

    if block_name != "all":
        stmt = stmt.where(
            and_(
                a_data.c.timepoint_s
                > (stmt_block.c.block_start_samples / 30000) - t_before,
                a_data.c.timepoint_s
                < (stmt_block.c.block_end_samples / 30000) + t_after,
            )
        )
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if signal_names:
        stmt = stmt.where(a_sigs.c.signal_name.in_(signal_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))

    with engine.connect() as conn:
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
        res.voltage = res.voltage.astype(np.float)
        res.timepoint_s = res.timepoint_s.astype(np.float)
    return res


def select_stft(
    engine,
    metadata,
    signal_names=None,
    block_name="pre",
    t_before=0,
    t_after=0,
    group_names=None,
    exp_names=None,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):
    stft, sesh_a_sig, a_sigs, r_sesh, rs_blocks, groups, experiments = (
        metadata.tables["analog_signal_stft"],
        metadata.tables["session_analog_signals"],
        metadata.tables["analog_signals"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )
    stmt_block = select(
        [
            rs_blocks.c.recording_session_id,
            rs_blocks.c.block_start_samples,
            rs_blocks.c.block_end_samples,
        ]
    )
    stmt_block = stmt_block.select_from(
        rs_blocks.join(r_sesh).join(groups).join(experiments)
    )
    if block_name != "all":
        stmt_block = stmt_block.where(rs_blocks.c.block_name == block_name)
    stmt_block = stmt_block.alias("block")
    columns = [
        stft.c.frequency,
        stft.c.fft_value,
        r_sesh.c.session_name,
        a_sigs.c.signal_name,
    ]
    if align_to_block:
        columns.append(
            (stft.c.timepoint_s - (stmt_block.c.block_start_samples / 30000)).label(
                "timepoint_s"
            )
        )
    else:
        columns.append(stft.c.timepoint_s)
    if group_names:
        columns.extend([groups.c.group_name])
    if exp_names:
        columns.extend([experiments.c.experiment_name, groups.c.group_name])

    stmt = select(columns)
    stmt = stmt.select_from(
        stft.join(sesh_a_sig)
        .join(a_sigs)
        .join(r_sesh, r_sesh.c.id == sesh_a_sig.c.recording_session_id)
        .join(rs_blocks)
        .join(groups)
        .join(experiments)
        .join(
            stmt_block, stmt_block.c.recording_session_id == r_sesh.c.id, isouter=False
        )
    )
    if block_name != "all":
        stmt = stmt.where(
            and_(
                stft.c.timepoint_s
                > (stmt_block.c.block_start_samples / 30000) - t_before,
                stft.c.timepoint_s < (stmt_block.c.block_end_samples / 30000) + t_after,
            )
        )
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if signal_names:
        stmt = stmt.where(a_sigs.c.signal_name.in_(signal_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))

    with engine.connect() as conn:
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
        res.frequency = res.frequency.astype(np.float)
        res.fft_value = res.fft_value.astype(np.float)
        res.timepoint_s = res.timepoint_s.astype(np.float)
    return res


def select_discrete_data(
    engine,
    metadata,
    signal_names=None,
    block_name="pre",
    t_before=0,
    t_after=0,
    group_names=None,
    exp_names=None,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):
    d_data, sesh_d_sig, d_sigs, r_sesh, rs_blocks, groups, experiments = (
        metadata.tables["discrete_signal_data"],
        metadata.tables["session_discrete_signals"],
        metadata.tables["discrete_signals"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    stmt_block = select(
        [
            rs_blocks.c.recording_session_id,
            rs_blocks.c.block_start_samples,
            rs_blocks.c.block_end_samples,
        ]
    )
    stmt_block = stmt_block.select_from(
        rs_blocks.join(r_sesh).join(groups).join(experiments)
    )
    if block_name != "all":
        stmt_block = stmt_block.where(rs_blocks.c.block_name == block_name)
    stmt_block = stmt_block.alias("block")

    columns = [
        d_sigs.c.signal_name,
        r_sesh.c.session_name,
    ]
    if align_to_block:
        columns.append(
            (d_data.c.timepoint_sample - stmt_block.c.block_start_samples).label(
                "timepoint_sample"
            )
        )
    else:
        columns.append(d_data.c.timepoint_sample)

    if group_names:
        columns.extend([groups.c.group_name])
    if exp_names:
        columns.extend([experiments.c.experiment_name, groups.c.group_name])

    stmt = select(columns)
    stmt = stmt.select_from(
        d_data.join(sesh_d_sig)
        .join(d_sigs)
        .join(r_sesh, r_sesh.c.id == sesh_d_sig.c.recording_session_id)
        .join(rs_blocks)
        .join(groups)
        .join(experiments)
        .join(
            stmt_block, stmt_block.c.recording_session_id == r_sesh.c.id, isouter=False
        )
    )

    if block_name != "all":
        stmt = stmt.where(
            and_(
                d_data.c.timepoint_sample
                > (stmt_block.c.block_start_samples - (t_before / 30000)),
                d_data.c.timepoint_sample
                < (stmt_block.c.block_end_samples + (t_after / 30000)),
            )
        )

    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if signal_names:
        stmt = stmt.where(d_sigs.c.signal_name.in_(signal_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))

    with engine.connect() as conn:
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
    return res


def _result_proxy_to_df(rp):
    data = rp.fetchall()
    return pd.DataFrame(data, columns=data[0].keys())
