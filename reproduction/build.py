from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import pandas as pd

from source_io import fetch

SOURCE_URL = 'https://www.fjc.gov/sites/default/files/idb/textfiles/cv88on.zip'
TARGET = 'trial_within_36m'
ID_COL = 'CASE_ID'
TIME_COL = 'FILEDATE'
FEATURES = [
    'NOS',
    'JURIS',
    'DISTRICT',
    'CIRCUIT',
    'JURY',
    'DEMANDED',
    'FILEDATE_YEAR_MONTH',
    'RESIDENC',
    'ARBIT',
]
OUT_COLUMNS = [ID_COL, TIME_COL] + FEATURES + [TARGET]
SPLITS = {
    'train': 'public/train.csv',
    'eval': 'public/eval.csv',
    'holdout': 'private/holdout.csv',
}
WINDOWS = {
    'train': ('2010-10-01', '2014-10-01'),
    'eval': ('2017-10-01', '2018-10-01'),
    'holdout': ('2021-10-01', '2022-10-01'),
}
DISTRICT_CODE_MASK = r'^(?:7-|[0-9]{1,3}[A-Z]?)$'
EXCLUDED_NOS_RANGES = ((460, 465), (510, 560), (861, 865))


def _norm(value):
    chars = []
    for ch in str(value).upper():
        chars.append(ch if (ch.isalnum() or ch == '_') else '_')
    return ''.join(chars).strip('_')


def _clean_text(series):
    return series.astype('string').str.strip().str.replace('  ', ' ', regex=False)


def _parse_num(series):
    value = series.astype('string').str.strip().str.replace(',', '', regex=False)
    value = value.replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA, 'NAN': pd.NA})
    return pd.to_numeric(value, errors='coerce')


def _parse_date(series):
    value = series.astype('string').str.strip()
    value = value.replace({'': pd.NA, 'nan': pd.NA, 'NaN': pd.NA, 'NAN': pd.NA})
    parsed = pd.to_datetime(value, format='%m/%d/%Y', errors='coerce')
    return parsed.mask(parsed == pd.Timestamp('1900-01-01'))


def _district_code(series):
    value = series.astype('string').str.strip().str.upper()
    value = value.replace({
        '': pd.NA,
        'nan': pd.NA,
        'NAN': pd.NA,
        'NONE': pd.NA,
        '-8': pd.NA,
        'UNKNOWN': pd.NA,
    })
    value = value.str.replace(r'^7\s*-+.*$', '7-', regex=True)
    value = value.str.replace(r'\s.*$', '', regex=True)
    value = value.str.replace(r'\s+', '', regex=True)
    return value.replace({'': pd.NA})


def _make_colmap(sample, has_header, member_name):
    ncols = sample.shape[1]

    def bad(message):
        text = (
            message
            + '; selected_archive_member: '
            + str(member_name)
            + '; observed_headers: '
            + repr([str(c) for c in sample.columns])[:2200]
            + '; ncols: '
            + str(ncols)
        )
        raise AssertionError(text[:4000])

    if has_header:
        norm_to_col = {}
        for col in sample.columns:
            key = _norm(col)
            if key and key not in norm_to_col:
                norm_to_col[key] = col
        expected = {
            'CIRCUIT': ['CIRCUIT'],
            'DISTRICT': ['DISTRICT'],
            'OFFICE': ['OFFICE'],
            'DOCKET': ['DOCKET'],
            'FILEDATE': ['FILEDATE'],
            'TERMDATE': ['TERMDATE'],
            'ORIGIN': ['ORIGIN'],
            'NOS': ['NOS'],
            'JURIS': ['JURIS'],
            'RESIDENC': ['RESIDENC', 'RESIDENCE'],
            'DEMANDED': ['DEMANDED'],
            'JURY': ['JURY'],
            'PROCPROG': ['PROCPROG', 'PROC_PROG'],
            'TAPEYEAR': ['TAPEYEAR', 'TAPE_YEAR'],
            'ARBIT': ['ARBIT', 'ARBITRATION'],
        }
        cmap = {}
        missing = []
        for key, aliases in expected.items():
            found = None
            for alias in aliases:
                candidate = _norm(alias)
                if candidate in norm_to_col:
                    found = norm_to_col[candidate]
                    break
            if found is None:
                missing.append(key)
            else:
                cmap[key] = found
        if missing:
            bad('missing source fields: ' + ', '.join(missing))
        return cmap

    def col(index):
        return sample.iloc[:, index].astype(str).str.strip()

    date_scores = []
    for index in range(ncols):
        values = col(index)
        score = float(((values.str.count('/') == 2) & (values.str.len() >= 8)).mean())
        if score >= 0.10:
            date_scores.append((index, score))
    date_scores.sort()
    if len(date_scores) < 2:
        bad('could not identify FILEDATE and TERMDATE date columns')
    filedate_i = date_scores[0][0]
    termdate_i = date_scores[1][0]

    def ok_range(index, low, high, fraction=0.85):
        values = _parse_num(col(index))
        if float(values.notna().mean()) < 0.05:
            return False
        present = values.dropna()
        if len(present) == 0:
            return False
        return float(present.between(low, high).mean()) >= fraction

    cmap = {
        'CIRCUIT': 0,
        'DISTRICT': 1,
        'OFFICE': 2,
        'DOCKET': 3,
        'FILEDATE': filedate_i,
        'TERMDATE': termdate_i,
        'ORIGIN': 8,
        'NOS': 9,
        'JURIS': 10,
        'RESIDENC': 11,
        'DEMANDED': 12,
        'JURY': 13,
        'PROCPROG': 15,
        'TAPEYEAR': 17,
    }

    if not (
        ok_range(8, 1, 9)
        and ok_range(9, 1, 999)
        and ok_range(10, 1, 9)
        and ok_range(15, 0, 9)
        and ok_range(17, 1988, 2026)
    ):
        exclude = {0, 1, 2, 3, filedate_i, termdate_i}

        def find_range(low, high, min_unique=1, max_unique=10000, exclude_set=None):
            skip = set() if exclude_set is None else set(exclude_set)
            best = None
            for index in range(ncols):
                if index in skip:
                    continue
                values = _parse_num(col(index))
                if float(values.notna().mean()) < 0.05:
                    continue
                present = values.dropna()
                if len(present) == 0:
                    continue
                if float(present.between(low, high).mean()) < 0.95:
                    continue
                unique_count = int(present.nunique())
                if unique_count < min_unique or unique_count > max_unique:
                    continue
                score = float(values.notna().mean())
                if best is None or score > best[0]:
                    best = (score, index)
            return None if best is None else best[1]

        origin_i = find_range(1, 1, 1, 1, exclude)
        if origin_i is None:
            bad('could not identify ORIGIN field')
        cmap['ORIGIN'] = origin_i
        exclude.add(origin_i)
        nos_i = find_range(100, 999, 20, 2000, exclude)
        if nos_i is None:
            bad('could not identify NOS field')
        cmap['NOS'] = nos_i
        exclude.add(nos_i)
        juris_i = find_range(1, 9, 2, 10, exclude)
        if juris_i is None:
            bad('could not identify JURIS field')
        cmap['JURIS'] = juris_i
        exclude.add(juris_i)
        proc_i = find_range(0, 9, 2, 10, exclude)
        if proc_i is None:
            bad('could not identify PROCPROG field')
        cmap['PROCPROG'] = proc_i
        exclude.add(proc_i)
        tape_i = find_range(1988, 2026, 5, 100, exclude)
        if tape_i is None:
            bad('could not identify TAPEYEAR field')
        cmap['TAPEYEAR'] = tape_i

    used = set(cmap.values())
    arbit_i = None
    best_score = None
    for index in range(ncols):
        if index in used:
            continue
        values = col(index).str.upper()
        present = [
            str(x).strip()
            for x in values.dropna().unique()
            if str(x).strip() not in ('', 'NAN', 'NONE')
        ]
        candidate_values = set(present)
        if not candidate_values or not candidate_values.issubset({'M', 'V', 'E', 'Y'}):
            continue
        coverage = float((values.str.strip() != '').mean())
        score = (len(candidate_values), coverage)
        if best_score is None or score > best_score:
            best_score = score
            arbit_i = index
    if arbit_i is None:
        for index in range(24, min(ncols, 36)):
            if index in used:
                continue
            values = col(index).str.upper()
            present = set(
                str(x).strip()
                for x in values.dropna().unique()
                if str(x).strip() not in ('', 'NAN', 'NONE')
            )
            if present and present.intersection({'M', 'V', 'E', 'Y'}) and present.issubset({'M', 'V', 'E', 'Y', 'N'}):
                arbit_i = index
                break
    if arbit_i is None:
        bad('could not identify ARBIT field')
    cmap['ARBIT'] = arbit_i
    return cmap


def build(output: Path, cache: Path):
    output = Path(output)
    cache = Path(cache)
    (output / 'public').mkdir(parents=True, exist_ok=True)
    (output / 'private').mkdir(parents=True, exist_ok=True)
    archive = Path(fetch(SOURCE_URL, cache, 'cv88on.zip'))
    checks = []

    def add_check(name, observed, expected, passed, kind=None):
        item = {
            'check': name,
            'observed': str(observed)[:1000],
            'expected': str(expected)[:1000],
            'passed': bool(passed),
        }
        if kind is not None:
            item['kind'] = kind
        checks.append(item)

    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.infolist() if not m.is_dir()]
        tabular = [m for m in members if m.filename.lower().endswith(('.txt', '.tsv', '.csv'))]
        preferred = [m for m in tabular if 'cv88on' in Path(m.filename).name.lower()]
        candidates = preferred or tabular
        if not candidates:
            raise AssertionError(
                (
                    'No tabular archive member found; members: '
                    + repr([m.filename for m in members])
                )[:4000]
            )
        member = sorted(candidates, key=lambda m: len(m.filename))[0]
        add_check('selected_archive_member', member.filename, 'one cv88on text member', bool(member.filename))

        with zf.open(member) as fh:
            first_line = fh.readline().decode('latin-1').rstrip()
        first_fields = [x.strip() for x in first_line.split('\t')]
        header_norms = {_norm(x) for x in first_fields}
        has_header = any(x in header_norms for x in ('CIRCUIT', 'DISTRICT', 'DOCKET', 'FILEDATE'))
        add_check(
            'header_detection',
            'header' if has_header else 'headerless',
            'detect a header row or assign codebook names to a headerless tab-delimited file',
            len(first_fields) > 3,
        )

        read_kwargs = dict(
            sep='\t',
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding='latin-1',
            on_bad_lines='skip',
        )
        if has_header:
            sample = pd.read_csv(zf.open(member), header=0, nrows=5000, **read_kwargs)
        else:
            sample = pd.read_csv(zf.open(member), header=None, nrows=5000, **read_kwargs)
        cmap = _make_colmap(sample, has_header, member.filename)

        def g(frame, key):
            return frame[cmap[key]]

        counters = {
            'source_rows': 0,
            'origin_counts': {},
            'valid_filedate_origin1': 0,
            'missing_filedate_origin1': 0,
            'missing_nos_origin1': 0,
            'missing_juris_origin1': 0,
            'missing_docket_origin1': 0,
            'missing_district_origin1': 0,
            'unparsed_district_origin1': 0,
            'hyphen_raw_origin1': 0,
            'hyphen_preserved_origin1': 0,
            'excluded_nos': {'460-465': 0, '510-560': 0, '861-865': 0},
        }
        district_counts = {}
        frames = []
        reader = pd.read_csv(
            zf.open(member),
            header=(0 if has_header else None),
            chunksize=150000,
            **read_kwargs,
        )
        for chunk in reader:
            counters['source_rows'] += len(chunk)
            origin = _parse_num(g(chunk, 'ORIGIN'))
            nos = _parse_num(g(chunk, 'NOS'))
            juris = _parse_num(g(chunk, 'JURIS'))
            filedate = _parse_date(g(chunk, 'FILEDATE'))
            termdate = _parse_date(g(chunk, 'TERMDATE'))
            docket = _clean_text(g(chunk, 'DOCKET')).replace({
                '': pd.NA,
                'nan': pd.NA,
                'NaN': pd.NA,
                'NAN': pd.NA,
            })
            raw_district = _clean_text(g(chunk, 'DISTRICT')).str.upper().str.strip()
            raw_district = raw_district.replace({
                '': pd.NA,
                'NAN': pd.NA,
                'NONE': pd.NA,
                '-8': pd.NA,
            })
            district = _district_code(g(chunk, 'DISTRICT'))
            district_valid = district.notna() & district.str.match(DISTRICT_CODE_MASK, na=False)
            origin1 = origin.eq(1)
            hyphen_raw = origin1 & raw_district.fillna('').str.contains('-', regex=False)

            for key, count in origin.value_counts(dropna=False).items():
                name = str(key)
                counters['origin_counts'][name] = counters['origin_counts'].get(name, 0) + int(count)
            counters['valid_filedate_origin1'] += int((origin1 & filedate.notna()).sum())
            counters['missing_filedate_origin1'] += int((origin1 & filedate.isna()).sum())
            counters['missing_nos_origin1'] += int((origin1 & nos.isna()).sum())
            counters['missing_juris_origin1'] += int((origin1 & juris.isna()).sum())
            counters['missing_docket_origin1'] += int((origin1 & docket.isna()).sum())
            counters['missing_district_origin1'] += int((origin1 & district.isna()).sum())
            counters['unparsed_district_origin1'] += int((origin1 & district.notna() & ~district_valid).sum())
            counters['hyphen_raw_origin1'] += int(hyphen_raw.sum())
            counters['hyphen_preserved_origin1'] += int(
                (hyphen_raw & district.fillna('').str.endswith('-')).sum()
            )
            counters['excluded_nos']['460-465'] += int((origin1 & nos.between(460, 465)).sum())
            counters['excluded_nos']['510-560'] += int((origin1 & nos.between(510, 560)).sum())
            counters['excluded_nos']['861-865'] += int((origin1 & nos.between(861, 865)).sum())

            counted = origin1 & district_valid
            if bool(counted.any()):
                for key, count in district[counted].value_counts().items():
                    label = str(key)
                    district_counts[label] = district_counts.get(label, 0) + int(count)

            in_windows = (
                ((filedate >= pd.Timestamp(WINDOWS['train'][0])) & (filedate < pd.Timestamp(WINDOWS['train'][1])))
                | ((filedate >= pd.Timestamp(WINDOWS['eval'][0])) & (filedate < pd.Timestamp(WINDOWS['eval'][1])))
                | ((filedate >= pd.Timestamp(WINDOWS['holdout'][0])) & (filedate < pd.Timestamp(WINDOWS['holdout'][1])))
            )
            keep = origin1 & filedate.notna() & nos.notna() & juris.notna() & docket.notna() & district_valid
            keep = keep & nos.between(1, 999) & juris.between(1, 5)
            keep = (
                keep
                & (~nos.between(460, 465))
                & (~nos.between(510, 560))
                & (~nos.between(861, 865))
            )
            keep = keep & in_windows
            if not bool(keep.any()):
                continue
            sub = pd.DataFrame({
                'CIRCUIT': _parse_num(g(chunk, 'CIRCUIT')),
                'DISTRICT': district,
                'OFFICE': _parse_num(g(chunk, 'OFFICE')),
                'DOCKET': docket,
                'FILEDATE': filedate,
                'TERMDATE': termdate,
                'NOS': nos,
                'JURIS': juris,
                'RESIDENC': _clean_text(g(chunk, 'RESIDENC')),
                'DEMANDED': _parse_num(g(chunk, 'DEMANDED')),
                'JURY': _clean_text(g(chunk, 'JURY')),
                'PROCPROG': _parse_num(g(chunk, 'PROCPROG')),
                'TAPEYEAR': _parse_num(g(chunk, 'TAPEYEAR')),
                'ARBIT': _clean_text(g(chunk, 'ARBIT')),
            })
            frames.append(sub.loc[keep].copy())

        add_check('malformed_rows_policy', 'malformed rows skipped by the delimited parser', 'drop malformed rows', True)
        add_check('source_rows', counters['source_rows'], 'positive row count', counters['source_rows'] > 0)
        origin1_count = sum(v for k, v in counters['origin_counts'].items() if k in ('1', '1.0'))
        add_check(
            'origin_rows',
            {'origin_equal_1': origin1_count, 'origin_counts': counters['origin_counts']},
            'at least one ORIGIN equal to 1 row',
            origin1_count > 0,
        )
        add_check(
            'valid_filedate_rows',
            counters['valid_filedate_origin1'],
            'at least one valid FILEDATE among ORIGIN equal to 1 rows',
            counters['valid_filedate_origin1'] > 0,
        )
        add_check(
            'missing_key_counts',
            {
                'FILEDATE': counters['missing_filedate_origin1'],
                'NOS': counters['missing_nos_origin1'],
                'JURIS': counters['missing_juris_origin1'],
                'DOCKET': counters['missing_docket_origin1'],
            },
            'missing key rows recorded and excluded from the population',
            True,
        )
        add_check(
            'excluded_nos_counts',
            counters['excluded_nos'],
            'excluded NOS ranges recorded; their presence elsewhere in the source does not abort the build',
            True,
        )

        letter_codes = sorted(k for k in district_counts if re.search(r'[A-Z]$', k))
        letter_rows = int(sum(district_counts[k] for k in letter_codes))
        hyphen_codes = sorted(k for k in district_counts if '-' in k)
        hyphen_rows = int(sum(district_counts[k] for k in hyphen_codes))
        hyphen_preserved_ok = (
            counters['hyphen_raw_origin1'] == 0
            or counters['hyphen_preserved_origin1'] == counters['hyphen_raw_origin1']
        )
        add_check(
            'district_code_parse',
            {
                'missing_district_rows_among_origin_one': counters['missing_district_origin1'],
                'unparsed_district_rows_among_origin_one': counters['unparsed_district_origin1'],
                'distinct_district_codes': len(district_counts),
                'letter_suffixed_codes': letter_codes[:20],
                'letter_suffixed_rows': letter_rows,
                'hyphen_suffixed_codes': hyphen_codes[:20],
                'hyphen_suffixed_rows': hyphen_rows,
                'raw_district_rows_with_hyphen': counters['hyphen_raw_origin1'],
                'preserved_hyphen_rows': counters['hyphen_preserved_origin1'],
            },
            'every ORIGIN equal to 1 row carries the codebook district code; the seven letter-suffixed districts such as 3A, 3C and 3N stay text instead of becoming missing numbers; and the Alaska code 7- is preserved rather than truncated to 7',
            bool(
                counters['missing_district_origin1'] == 0
                and counters['unparsed_district_origin1'] == 0
                and letter_rows > 0
                and hyphen_preserved_ok
            ),
        )

        if not frames:
            add_check('cohort_rows', 0, 'at least one row in the fixed FILEDATE windows', False)
            failed = [c for c in checks if not c['passed']]
            raise AssertionError(('Source checks failed: ' + json.dumps(failed, ensure_ascii=False))[:4000])

        df = pd.concat(frames, ignore_index=True)
        df['DISTRICT'] = df['DISTRICT'].astype('string')
        df['_TAPEYEAR'] = _parse_num(df['TAPEYEAR']).fillna(0)
        df = df.sort_values(
            ['CIRCUIT', 'DISTRICT', 'OFFICE', 'DOCKET', '_TAPEYEAR', 'FILEDATE'],
            kind='mergesort',
            na_position='first',
        )
        pre_dedup_rows = int(len(df))
        duplicate_key_rows = int(
            df.duplicated(subset=['CIRCUIT', 'DISTRICT', 'OFFICE', 'DOCKET']).sum()
        )
        df = df.drop_duplicates(subset=['CIRCUIT', 'DISTRICT', 'OFFICE', 'DOCKET'], keep='last')
        df = df.drop(columns=['_TAPEYEAR']).reset_index(drop=True)
        post_dedup_rows = int(len(df))
        add_check(
            'dedup_key_duplicates',
            {
                'cohort_rows_before_dedup': pre_dedup_rows,
                'duplicate_key_rows': duplicate_key_rows,
                'rows_after_dedup': post_dedup_rows,
            },
            'duplicate docket keys are removed under the documented maximum-TAPEYEAR rule, and the district is part of the key as its text code so distinct letter-coded and hyphen-coded districts cannot merge',
            (pre_dedup_rows - post_dedup_rows) == duplicate_key_rows,
        )

        def split_of(value):
            if pd.isna(value):
                return None
            for name, window in WINDOWS.items():
                if pd.Timestamp(window[0]) <= value < pd.Timestamp(window[1]):
                    return name
            return None

        df['_split'] = df['FILEDATE'].map(split_of)
        df = df[df['_split'].notna()].copy().reset_index(drop=True)

        def znum(series, width):
            value = _parse_num(series).round()
            value = value.mask(value < 0)
            return value.astype('Int64').astype('string').fillna('MISSING').str.zfill(width)

        docket_id = df['DOCKET'].astype('string').str.strip().str.replace(' ', '', regex=False).fillna('MISSING')
        district_txt = df['DISTRICT'].astype('string').fillna('MISSING')
        case_id = (
            znum(df['CIRCUIT'], 2)
            + '-'
            + district_txt
            + '-'
            + znum(df['OFFICE'], 1)
            + '-'
            + docket_id
        )

        jury = _clean_text(df['JURY']).str.upper().replace({
            '': pd.NA,
            'NAN': pd.NA,
            'NONE': pd.NA,
            '-8': pd.NA,
        }).fillna('MISSING')
        demand_num = _parse_num(df['DEMANDED'])
        demand_num = demand_num.mask(demand_num < 0)
        demand = demand_num.round().astype('Int64').astype('string').fillna('MISSING')
        resid_raw = _clean_text(df['RESIDENC']).str.upper().replace({
            '': pd.NA,
            'NAN': pd.NA,
            'NONE': pd.NA,
            '-8': pd.NA,
        })
        resid_num = _parse_num(resid_raw)
        resid = resid_num.round().astype('Int64').astype('string').str.zfill(2).fillna('MISSING')
        arbit = _clean_text(df['ARBIT']).str.upper().replace({
            '': pd.NA,
            'NAN': pd.NA,
            'NONE': pd.NA,
            '-8': pd.NA,
        }).fillna('MISSING')
        filedate_month = df['FILEDATE'].dt.strftime('%Y-%m')

        proc = _parse_num(df['PROCPROG'])
        term = df['TERMDATE']
        horizon = df['FILEDATE'] + pd.DateOffset(months=36)
        label = (proc.isin([6, 7, 8, 9]) & term.notna() & term.le(horizon)).astype(int)

        out = pd.DataFrame({
            'CASE_ID': case_id,
            'FILEDATE': df['FILEDATE'].dt.strftime('%Y-%m-%d'),
            'NOS': znum(df['NOS'], 3),
            'JURIS': znum(df['JURIS'], 1),
            'DISTRICT': district_txt,
            'CIRCUIT': znum(df['CIRCUIT'], 2),
            'JURY': jury,
            'DEMANDED': demand,
            'FILEDATE_YEAR_MONTH': filedate_month,
            'RESIDENC': resid,
            'ARBIT': arbit,
            TARGET: label,
        })
        out['_split'] = df['_split'].values
        out = out[OUT_COLUMNS + ['_split']]

        coverage = {
            name: int(out[name].astype('string').ne('MISSING').sum())
            for name in FEATURES
        }
        add_check(
            'feature_coverage',
            coverage,
            'each declared feature has at least one nonmissing value',
            all(v > 0 for v in coverage.values()),
        )

        positives = int(out[TARGET].sum())
        add_check('target_positive_rows', positives, 'at least one positive trial outcome', positives > 0)
        positives_by_split = {
            name: int(out.loc[out['_split'] == name, TARGET].sum())
            for name in WINDOWS
        }
        add_check(
            'positive_events_by_split',
            positives_by_split,
            'each fixed split has at least one positive event',
            all(v > 0 for v in positives_by_split.values()),
        )

        observed_windows = {}
        window_pass = {}
        for name, window in WINDOWS.items():
            frame = out[out['_split'] == name]
            if frame.empty:
                observed_windows[name] = {'min': None, 'max': None}
                window_pass[name] = False
            else:
                values = pd.to_datetime(frame['FILEDATE'])
                observed_windows[name] = {
                    'min': str(values.min().date()),
                    'max': str(values.max().date()),
                }
                window_pass[name] = bool(
                    values.ge(pd.Timestamp(window[0])).all()
                    and values.lt(pd.Timestamp(window[1])).all()
                )
        add_check(
            'split_windows',
            observed_windows,
            {k: {'start': v[0], 'end_exclusive': v[1]} for k, v in WINDOWS.items()},
            all(window_pass.values()),
        )

        duplicate_count = int(out['CASE_ID'].duplicated().sum())
        add_check('duplicate_case_ids', duplicate_count, 0, duplicate_count == 0)

        split_frames = {}
        for name in WINDOWS:
            frame = out[out['_split'] == name].copy()
            frame = frame[OUT_COLUMNS]
            frame = frame.sort_values(['FILEDATE', 'CASE_ID'], kind='mergesort').reset_index(drop=True)
            split_frames[name] = frame
        add_check(
            'split_row_counts',
            {k: len(v) for k, v in split_frames.items()},
            'nonempty train, eval and holdout frames',
            all(len(v) > 0 for v in split_frames.values()),
        )
        observed_columns = {k: list(v.columns) for k, v in split_frames.items()}
        add_check(
            'identical_columns',
            observed_columns,
            OUT_COLUMNS,
            all(v.columns.tolist() == OUT_COLUMNS for v in split_frames.values()),
        )
        add_check(
            'target_values',
            sorted(int(x) for x in out[TARGET].unique()),
            [0, 1],
            set(int(x) for x in out[TARGET].unique()).issubset({0, 1}),
        )

        demand_values = demand_num.dropna()
        demand_observed = {
            'nonmissing': int(len(demand_values)),
            'min': float(demand_values.min()) if len(demand_values) else None,
            'max': float(demand_values.max()) if len(demand_values) else None,
            'top_code_9999_count': int((demand_values == 9999).sum()),
        }
        demand_pass = bool(
            len(demand_values) > 0
            and float(demand_values.min()) >= 0
            and float(demand_values.max()) <= 9999
        )
        add_check(
            'demanded_units',
            demand_observed,
            'nonnegative thousands of dollars, top-coded at 9999',
            demand_pass,
            kind='unit',
        )

        positive_mask = label.eq(1)
        if bool(positive_mask.any()):
            elapsed_days = (df.loc[positive_mask, 'TERMDATE'] - df.loc[positive_mask, 'FILEDATE']).dt.days
            max_days = int(elapsed_days.max())
        else:
            max_days = None
        add_check(
            'trial_horizon_days',
            max_days,
            'no more than 1097 days, the 36-month horizon',
            max_days is not None and max_days <= 1097,
            kind='unit',
        )

        rule_matches = int((proc.isin([6, 7, 8, 9]) & term.notna() & term.le(horizon)).sum())
        add_check(
            'label_rule',
            {'positives': positives, 'rule_matches': rule_matches},
            'all positives are PROCPROG 6 through 9 with TERMDATE on or before FILEDATE plus 36 months',
            rule_matches == positives,
        )

        failed = [c for c in checks if not c['passed']]
        if failed:
            raise AssertionError(('Source checks failed: ' + json.dumps(failed, ensure_ascii=False))[:4000])

        for name, relative in SPLITS.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            split_frames[name].to_csv(path, index=False)
        return checks
