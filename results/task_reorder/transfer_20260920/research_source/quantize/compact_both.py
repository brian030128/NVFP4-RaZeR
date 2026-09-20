"""Compact both permutations without changing original-coordinate format choices.

Columns move only as intact16-element scale groups. No loss values are used.
"""
import copy
import torch
from quantize.compact_rows import compact_rows


def compact_both(layout):
    cfg = layout['config']; n,k = layout['weight_shape']
    assert cfg['atom_rows'] == 1 and cfg['atom_cols'] == 16
    assert cfg['tile_cols'] == 64 and cfg['axes'] == 'both'
    old_columns = layout['col_perm'].cpu()
    column_groups = old_columns.reshape(-1,16)[:,0] // 16
    assert torch.equal(old_columns, (column_groups[:,None]*16 + torch.arange(16)).flatten())
    assert torch.equal(column_groups.sort().values,torch.arange(k//16))
    # Row compaction sees a fixed column basis. Temporarily hide that basis from
    # its row-only API; restore it before constructing the final layout.
    proxy = copy.deepcopy(layout)
    proxy['config']['axes'] = 'rows';proxy['col_perm'] = torch.arange(k)
    result = compact_rows(proxy)
    result['config'] = copy.deepcopy(cfg)
    result['col_perm'] = old_columns.clone()
    # Transpose the type grid and treat each16-column scale group as one row.
    # Four such groups constitute one64-column type tile.
    transposed = dict(config=dict(axes='rows',atom_rows=1,tile_rows=4,tile_cols=1),
        weight_shape=[k//16,result['mask'].shape[0]],
        row_perm=column_groups, inverse_row_perm=column_groups.argsort(), row_atom_perm=column_groups,
        col_perm=torch.arange(result['mask'].shape[0]),mask=result['mask'].T.contiguous())
    column_result = compact_rows(transposed)
    groups = column_result['row_perm']
    result['col_perm'] = (groups[:,None]*16 + torch.arange(16)).flatten()
    result['inverse_col_perm'] = result['col_perm'].argsort()
    result['col_atom_perm'] = groups.clone()
    result['mask'] = column_result['mask'].T.contiguous()
    result['column_compaction'] = dict(method='active64-column-group assignment preserving16-column scales',
        old_scale_groups_moved=int((column_groups != torch.arange(k//16)).sum()),
        new_scale_groups_moved=int((groups != torch.arange(k//16)).sum()),
        old_columns_moved=int((old_columns != torch.arange(k)).sum()),
        new_columns_moved=int((result['col_perm'] != torch.arange(k)).sum()),
        active_column_groups=int(layout['mask'].any(0).sum()), native_latency_measured=False)
    assert result['column_compaction']['new_columns_moved'] <= result['column_compaction']['old_columns_moved']
    assert int(result['mask'].sum()) == int(layout['mask'].sum())
    # Compare original-coordinate type decisions at16-column-group resolution.
    def types(value):
        return value['mask'].repeat_interleave(cfg['tile_rows'],0).repeat_interleave(4,1)[
            value['inverse_row_perm']][:,value['col_atom_perm'].argsort()]
    assert torch.equal(types(result),types(layout))
    return result
