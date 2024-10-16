import numpy as np

def merge_intervals(arr, offset=0.35):
    # 各要素の(-offset, +offset)の範囲を計算
    intervals = np.array([(x - offset, x + offset) for x in arr])

    # 範囲を開始順にソート
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]

    # 初期化
    merged_intervals = np.empty_like(sorted_intervals)
    merged_intervals[0] = sorted_intervals[0]
    merged_count = 1

    for next_interval in sorted_intervals[1:]:
        current_interval = merged_intervals[merged_count - 1]

        # 最大値を求めることで、重なりを考慮
        start_max = np.maximum(current_interval[0], next_interval[0])
        end_min = np.minimum(current_interval[1], next_interval[1])

        # 重なりのチェックを直接して、結合を選択
        overlap_exists = start_max <= end_min

        # 重ならない場合、current_intervalを更新
        merged_intervals[merged_count - 1] = overlap_exists * np.array([np.minimum(current_interval[0], next_interval[0]), 
                                                                        np.maximum(current_interval[1], next_interval[1])]) \
                                             + (1 - overlap_exists) * current_interval

        # 重ならない場合、新しいインターバルを追加
        merged_count += (1 - overlap_exists)
        merged_intervals[merged_count - 1] = next_interval * (1 - overlap_exists) + overlap_exists * merged_intervals[merged_count - 1]

    return merged_intervals[:merged_count]


def _merge_intervals(arr, offset=100):
    # 各要素の(-0.5, +0.5)の範囲を計算
    intervals = np.array([(x - offset, x + offset) for x in arr])

    # 範囲を開始順にソート
    sorted_intervals = intervals[np.argsort(intervals[:, 0])]

    merged_intervals = []
    current_interval = sorted_intervals[0]
    for next_interval in sorted_intervals[1:]:
        # 最大値を求めることで、重なりを考慮
        start_max = np.maximum(current_interval[0], next_interval[0])
        end_min = np.minimum(current_interval[1], next_interval[1])

        # 重なりのチェックを直接して、結合を選択
        overlap_exists = start_max <= end_min

        # 重ならない場合はcurrent_intervalを結果に追加
        merged_intervals += [current_interval] * (1 - overlap_exists)
        
        # 重ならない場合はcurrent_intervalを更新
        current_interval = overlap_exists * np.array([np.minimum(current_interval[0], next_interval[0]), 
                                                      np.maximum(current_interval[1], next_interval[1])]) \
                           + (1 - overlap_exists) * next_interval

    # 最後のintervalを追加
    merged_intervals.append(current_interval)

    return np.array(merged_intervals)

test_cases = [
    np.array([1.0, 2.1, 2.8, 4.5]),  # Normal case with overlaps
    np.array([1.5, 2.0, 2.5, 1.0]),  # All overlap
    np.array([1.0, 3.0, 5.0]),        # No overlap
    np.array([1.0]),                  # Single element
    np.array([0.0, 0.5, 1.0, 1.5, 2.0]), # Overlaps every alternate element
    np.array([1.0, 1.4, 2.0, 2.4, 3.0]), # Overlaps between some, not all
    np.array([-2.0, -1.5, 0.0, 2.0]), # Negative numbers
    np.array([1.0, 1.0, 1.0]),        # Duplicate elements
    np.array([10.0, 9.5, 9.0, 8.5]),  # Reverse ordered input
    np.array([100.0, 200.0, 300.0]),  # Large gaps between elements
]
for i in range(len(test_cases)):
    result = merge_intervals(test_cases[i])
    print("case: \n",test_cases[i].ravel(),"\n result: \n",result)
    print("--------------------")
