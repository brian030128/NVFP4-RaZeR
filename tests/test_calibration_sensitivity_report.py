"""Check that exported tables retain every dataset, counts, and unrounded means."""
from summarize_calibration_sensitivity import DOMAINS, POLICIES, table


def main():
    r = dict(block_statistics={}, evaluation={})
    for i, policy in enumerate(POLICIES):
        r['block_statistics'][policy] = dict(selected_blocks=i, selected_fraction=i/10000,
                                            calibration_sequences=64)
        r['evaluation'][policy] = {d: dict(ppl=1.1234567 + j + i*.1) for j,d in enumerate(DOMAINS)}
    absolute = table('qwen4b', r)
    delta = table('qwen4b', r, delta=True)
    rows = [line for line in absolute if line.startswith('| ')][1:]
    delta_rows = [line for line in delta if line.startswith('| ')][1:]
    assert len(rows) == len(delta_rows) == len(POLICIES)
    for i, (row, change) in enumerate(zip(rows, delta_rows)):
        fields = [f.strip() for f in row.split('|')[1:-1]]
        changes = [f.strip() for f in change.split('|')[1:-1]]
        assert fields[0] == POLICIES[i] and int(fields[2]) == i
        assert fields[4:9] == [f'{1.1234567+j+i*.1:.6f}' for j in range(5)]
        assert fields[9] == f'{3.1234567+i*.1:.6f}'
        assert changes[4:10] == [f'{i*.1:+.6f}'] * 6
    print('Report preserves all five dataset values, E0M3 counts, and means from unrounded PPL')


if __name__ == '__main__':
    main()
