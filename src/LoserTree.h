#ifndef LOSERTREE_H
#define LOSERTREE_H

#include <cstdint>
#include <functional>

/**
 * 败者树
 * @tparam T         叶子元素类型
 * @tparam Ways      最大路数
 * @tparam Compare   比较器
 */
template <typename T, int Ways, typename Compare = std::less<T>>
class LoserTree {
public:
    static constexpr int K = Ways;

private:
    Compare comp;
    int loser_tree[2 * K];           // [0..K-1] 内部节点, [K..2K-1] 叶子存路索引
    T SENTINEL;                      // 路耗尽标记
    int num_active_ways;

public:
    T leaf_elements[K];              // 每个叶子当前的元素
    int total_elems;
private:

    int ptr[K];                      // 每路当前读取位置
    int end[K];                      // 每路结束位置
    T** chunks;                      // 指向各路 T 数组的指针

public:
    explicit LoserTree(const T& sentinel) : SENTINEL(sentinel) {
        total_elems = 0;
        num_active_ways = 0;
        chunks = nullptr;
    }

    /**
     * 初始化败者树。
     *
     * @param chunk_ptrs   T 数组指针
     * @param lengths     各路长度
     * @param num_blocks  实际活跃路数
     * @return 叶子索引
     */
    int init(T** chunk_ptrs, int* lengths, int num_blocks) {
        num_active_ways = num_blocks;
        chunks = chunk_ptrs;
        total_elems = 0;

        for (int i = 0; i < num_active_ways; i++) {
            if (lengths[i] > 0) {
                ptr[i] = 0;
                end[i] = lengths[i];
                total_elems += lengths[i];
            } else {
                ptr[i] = -1;
                end[i] = -1;
            }
        }
        for (int i = num_active_ways; i < Ways; i++) {
            ptr[i] = -1;
            end[i] = -1;
        }

        for (int i = 0; i < Ways; i++) {
            leaf_elements[i] = (ptr[i] != -1) ? chunks[i][0] : SENTINEL;
            loser_tree[K + i] = i;
        }

        return build_loser_tree();
    }

    int get_total() const { return total_elems; }

    /**
     * 推进 winner 所在路，返回新的全局胜者。
     */
    int advance(int winner_idx) {
        if (ptr[winner_idx] != -1 && ptr[winner_idx] + 1 < end[winner_idx]) {
            ptr[winner_idx]++;
            leaf_elements[winner_idx] = chunks[winner_idx][ptr[winner_idx]];
        } else {
            leaf_elements[winner_idx] = SENTINEL;
        }
        return replay_loser_tree(winner_idx);
    }

private:
    /**
     * 自底向上构建败者树。
     * 每个内部节点只比较左右子树的胜者，败者存入 loser_tree[p]，
     * 胜者沿临时数组 winners[] 向上传递。
     */
    int build_loser_tree() {
        int winners[2 * K];
        for (int i = 0; i < Ways; i++) {
            winners[K + i] = i;
        }

        for (int p = K - 1; p > 0; p--) {
            int left_winner  = winners[2 * p];
            int right_winner = winners[2 * p + 1];

            if (comp(leaf_elements[left_winner], leaf_elements[right_winner])) {
                loser_tree[p] = right_winner;  // 右败
                winners[p]    = left_winner;   // 左胜
            } else {
                loser_tree[p] = left_winner;   // 左败
                winners[p]    = right_winner;  // 右胜
            }
        }
        return winners[1];
    }

    /**
     * 更新后的叶子向上重新比赛，返回全局胜者。
     */
    int replay_loser_tree(int leaf_idx) {
        int p = (K + leaf_idx) >> 1;
        int competitor = leaf_idx;

        while (p > 0) {
            int loser_in_node = loser_tree[p];
            if (!comp(leaf_elements[competitor], leaf_elements[loser_in_node])) {
                // competitor 败 (比较器认为 competitor >= loser)
                loser_tree[p] = competitor;
                competitor = loser_in_node;
            }
            // competitor 胜 → 无需操作
            p >>= 1;
        }
        return competitor;
    }
};

#endif