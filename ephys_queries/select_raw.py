from pathlib import Path
from sqlalchemy import select


def get_raw_path(engine, metadata, session_name):
    """
    returns the path to raw data for a session given its name
    """
    r_sesh, groups, exp, exp_paths = (
        metadata.tables["recording_sessions"],
        metadata.tables["experimental_groups"],
        metadata.tables["experiments"],
        metadata.tables["experimental_paths"],
    )

    stmt = select([exp_paths.c.path_value])
    stmt = stmt.select_from(exp_paths.join(exp).join(groups).join(r_sesh))
    stmt = stmt.where(r_sesh.c.session_name == session_name)

    stmt_root = stmt.where(exp_paths.c.path_type == "exp_home_dir")
    stmt_dat = stmt.where(exp_paths.c.path_type == "dat_file_dir")

    with engine.connect() as con:
        rp_root = con.execute(stmt_root)
        rp_dat =  con.execute(stmt_dat)

    rp_root = rp_root.first()[0]
    rp_dat = rp_dat.first()[0]
    path = Path(rp_root) / rp_dat / session_name / f"{session_name}.dat"
    return path

