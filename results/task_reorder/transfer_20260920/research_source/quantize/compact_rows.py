"""Keep more rows fixed while preserving a frozen row-only mixed-format model.

Active 256-row sets remain intact; their physical tile positions can change.
All-E2M1 row groups can be refilled freely. No calibration losses are used.
"""
import copy
import torch


def maximum_assignment(weights):
    """Exact rectangular Hungarian assignment, integer weights, rows <= columns."""
    n = len(weights)
    if not n:
        return []
    m = len(weights[0])
    assert n <= m and all(len(row) == m for row in weights)
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    u = [0] * (n + 1); v = [0] * (m + 1)
    p = [0] * (m + 1); way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i; j0 = 0
        minimum = [float('inf')] * (m + 1); used = [False] * (m + 1)
        while True:
            used[j0] = True; i0 = p[j0]; delta = float('inf'); j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    value = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if value < minimum[j]:
                        minimum[j] = value; way[j] = j0
                    if minimum[j] < delta:
                        delta = minimum[j]; j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            assignment[p[j] - 1] = j - 1
    assert min(assignment) >= 0 and len(set(assignment)) == n
    return assignment


def original_row_types(layout):
    height = layout['config']['tile_rows']
    return layout['mask'].cpu().repeat_interleave(height, 0)[layout['inverse_row_perm'].cpu()]


def compact_rows(layout):
    config = layout['config']; n, k = layout['weight_shape']; height = config['tile_rows']
    assert config['atom_rows'] == 1 and config['axes'] == 'rows' and n % height == 0
    assert torch.equal(layout['col_perm'].cpu(), torch.arange(k)), 'Column permutations are outside this transform'
    old = layout['row_perm'].cpu()
    assert torch.equal(old.sort().values, torch.arange(n))
    assert torch.equal(layout['inverse_row_perm'].cpu(), torch.argsort(old))
    mask = layout['mask'].cpu()
    assert mask.dtype == torch.bool and mask.shape[0] == n // height
    groups = torch.where(mask.any(1))[0].tolist()
    row_sets = [old[g * height:(g + 1) * height].tolist() for g in groups]
    active_rows = {row for rows in row_sets for row in rows}
    total_in_block = [sum(row in active_rows for row in range(g * height, (g + 1) * height))
                      for g in range(n // height)]
    # Inactive rows can stay fixed unless their original slot is occupied by an
    # active block. Active rows stay fixed exactly at their group's overlap with
    # its destination. The two overlap terms below therefore maximize fixed rows
    # among assignments preserving each active group's original row membership.
    weights = [[total_in_block[g] + sum(row // height == g for row in rows)
                for g in range(n // height)] for rows in row_sets]
    targets = maximum_assignment(weights)
    perm = [-1] * n; new_mask = torch.zeros_like(mask)
    for group, rows, target in zip(groups, row_sets, targets):
        positions = list(range(target * height, (target + 1) * height))
        fixed = {row for row in rows if row // height == target}
        for row in fixed:
            perm[row] = row
        for position, row in zip([p for p in positions if p not in fixed], sorted(set(rows) - fixed)):
            perm[position] = row
        new_mask[target] = mask[group]
    for position in range(n):
        if perm[position] < 0 and position not in active_rows:
            perm[position] = position
    remaining = sorted(set(range(n)) - {row for row in perm if row >= 0})
    for position, row in zip([p for p, row in enumerate(perm) if row < 0], remaining):
        perm[position] = row
    rp = torch.tensor(perm, dtype=torch.long)
    result = copy.deepcopy(layout)
    result.update(row_perm=rp, inverse_row_perm=torch.argsort(rp), row_atom_perm=rp.clone(), mask=new_mask)
    assert torch.equal(rp.sort().values, torch.arange(n))
    assert torch.equal(original_row_types(result), original_row_types(layout)), 'Quantized model changed'
    assert int(result['mask'].sum()) == int(mask.sum())
    # Search derivatives refer to the old grouping, especially inactive rows.
    # Preserve the original file externally, not stale coordinate fields here.
    for key in ('fit_mask', 'election_upper_ce', 'election_upper_kl', 'proposal_upper_ce',
                'proposal_upper_kl', 'trace', 'fit_objective', 'identity_objective'):
        result.pop(key, None)
    result['row_compaction'] = dict(method='exact active-group assignment maximizing fixed rows',
        original_coordinate_type_map_identical=True, old_rows_moved=int((old != torch.arange(n)).sum()),
        new_rows_moved=int((rp != torch.arange(n)).sum()), active_row_groups=len(groups),
        source_active_groups=groups, destination_active_groups=targets,
        native_latency_measured=False)
    assert result['row_compaction']['new_rows_moved'] <= result['row_compaction']['old_rows_moved']
    return result
