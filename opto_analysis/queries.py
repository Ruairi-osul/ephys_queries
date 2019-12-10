import pandas as pd
from sqlalchemy.sql import select, cast, and_, or_
from sqlalchemy.types import Integer


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
    t_before=None,
    t_after=None,
    neuron_ids=None,
    group_names=None,
    exp_names=None,
    session_names=None,
    exclude_mua=True,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):

    ifr, neurons, r_sesh, rs_blocks, rs_config, groups, experiments = (
        metadata.tables["neuron_ifr"],
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["recording_session_config"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    sampling_rate = (
        select(
            [
                rs_config.c.recording_session_id,
                cast(rs_config.c.config_value, Integer()).label("sampling_rate"),
            ]
        )
        .where(rs_config.c.config == "sampleing_rate")
        .alias("sampling_rate")
    )

    block_t = (
        select([rs_blocks]).where(rs_blocks.c.block_name == block_name).alias("block_t")
    )

    columns: list = [ifr.c.neuron_id, ifr.c.ifr]
    if align_to_block:
        columns.extend(
            [
                (
                    ifr.c.timepoint_s
                    - (block_t.c.block_start_samples / sampling_rate.c.sampling_rate)
                ).label("timepoint_s")
            ]
        )
    else:
        columns.extend([ifr.c.timepoint_s])
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
        .join(sampling_rate, sampling_rate.c.recording_session_id == r_sesh.c.id)
    )

    stmt = stmt.where(rs_blocks.c.block_name == block_name)

    if block_name != "all":
        stmt = stmt.where(
            and_(
                ifr.c.timepoint_s
                > (rs_blocks.c.block_start_samples / sampling_rate.c.sampling_rate),
                ifr.c.timepoint_s
                < (rs_blocks.c.block_end_samples / sampling_rate.c.sampling_rate),
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
    spike_times, neurons, r_sesh, rs_blocks, rs_config, groups, experiments = (
        metadata.tables["spike_times"],
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["recording_session_config"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    sampling_rate = (
        select(
            [
                rs_config.c.recording_session_id,
                cast(rs_config.c.config_value, Integer()).label("sampling_rate"),
            ]
        )
        .where(rs_config.c.config == "sampleing_rate")
        .alias("sampling_rate")
    )

    block_t = (
        select([rs_blocks]).where(rs_blocks.c.block_name == block_name).alias("block_t")
    )

    columns: list = [spike_times.c.neuron_id]
    if align_to_block:
        columns.extend(
            [
                (
                    spike_times.c.spike_time_samples - block_t.c.block_start_samples
                ).label("spike_time_samples")
            ]
        )
    else:
        columns.extend([spike_times.c.spike_time_samples])
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
        .join(sampling_rate, sampling_rate.c.recording_session_id == r_sesh.c.id)
        .outerjoin(block_t)
    )

    if block_name != "all":
        stmt = stmt.where(
            and_(
                spike_times.c.spike_time_samples
                > (
                    block_t.c.block_start_samples
                    - (t_before * sampling_rate.c.sampling_rate)
                ),
                spike_times.c.spike_time_samples
                < block_t.c.block_end_samples
                + (t_after * sampling_rate.c.sampling_rate),
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
    waveforms, neurons, r_sesh, rs_config, groups, experiments = (
        metadata.tables["waveforms"],
        metadata.tables["neurons"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_config"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    sampling_rate = (
        select(
            [
                rs_config.c.recording_session_id,
                cast(rs_config.c.config_value, Integer()).label("sampling_rate"),
            ]
        )
        .where(rs_config.c.config == "sampleing_rate")
        .alias("sampling_rate")
    )

    columns = [waveforms.c.neuron_id, waveforms.c.waveform_value]
    if units == "seconds":
        columns.append(waveforms.c.waveform_index / sampling_rate.c.sampling_rate)
    elif units == "samples":
        columns.append(waveforms.c.waveform_value)
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
    stmt.select_from(
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
    session_names=None,
    as_df=True,
    align_to_block=False,
    exclude_excluded_recordings=True,
):

    a_data, sesh_a_sig, a_sigs, r_sesh, rs_config, rs_blocks, groups, experiments = (
        metadata.tables["analog_data"],
        metadata.tables["session_analog_signals"],
        metadata.tables["analog_signals"],
        metadata.tables["recording_sessions"],
        metadata.tables["recording_session_config"],
        metadata.tables["recording_session_block_times"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
    )

    sampling_rate = (
        select(
            [
                rs_config.c.recording_session_id,
                cast(rs_config.c.config_value, Integer()).label("sampling_rate"),
            ]
        )
        .where(rs_config.c.config == "sampleing_rate")
        .alias("sampling_rate")
    )

    block_t = (
        select([rs_blocks]).where(rs_blocks.c.block_name == block_name).alias("block_t")
    )

    columns = [
        a_data.c.voltage,
        r_sesh.c.session_name,
        a_sigs.c.signal_name,
    ]

    if align_to_block:
        columns.extend(
            [
                (
                    a_data.c.timepoint_s
                    - (block_t.c.block_start_samples / sampling_rate.c.sampling_rate)
                ).label("timepoint_s")
            ]
        )
    else:
        columns.extend([a_data.c.timepoint_s])

    stmt = select(columns)
    stmt = stmt.select_from(
        a_data.join(sesh_a_sig)
        .join(a_sigs)
        .join(r_sesh)
        .join(rs_blocks)
        .join(groups)
        .join(experiments)
        .join(sampling_rate, sampling_rate.c.recording_session_id == r_sesh.c.id)
        .outerjoin(block_t)
    )

    if block_name != "all":
        stmt = stmt.where(
            and_(
                a_data.c.timepoint_s
                > (
                    (block_t.c.block_start_samples / sampling_rate.c.sampling_rate)
                    - t_before
                ),
                a_data.c.timepoint_s
                < (block_t.c.block_start_samples / sampling_rate.c.sampling_rate)
                + t_after,
            )
        )
    if exclude_excluded_recordings:
        stmt = stmt.where(or_(r_sesh.c.excluded.is_(None), r_sesh.c.excluded == 1))
    if signal_names:
        stmt = stmt.where(a_sigs.c.signal_name.in_(signal_names))
    if session_names:
        stmt = stmt.where(r_sesh.c.session_name.in_(session_names))
    if group_names:
        stmt = stmt.where(groups.c.group_name.in_(group_names))
    if exp_names:
        stmt = stmt.where(experiments.c.experiment_name.in_(exp_names))

    with engine.connect() as conn:
        print(str(stmt))
        res = conn.execute(stmt)
    if as_df:
        res = _result_proxy_to_df(res)
    return res


def _result_proxy_to_df(rp):
    data = rp.fetchall()
    return pd.DataFrame(data, columns=data[0].keys())
