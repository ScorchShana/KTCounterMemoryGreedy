#ifndef FIXED_MIN_HEAP_HEADER
#define FIXED_MIN_HEAP_HEADER

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <algorithm>

template <typename T, std::size_t Capacity>
class FixedMinHeap {
public:
	constexpr FixedMinHeap() noexcept = default;

	bool push(const T& value) noexcept(kNoThrowCopyAssign && kNoThrowMove && kNoThrowLess) {
		if (full()) {
			return false;
		}
		const std::size_t index = ++size_;
		data_[index] = value;
		swim(index);
		return true;
	}

	bool pop() noexcept(kNoThrowMove && kNoThrowLess) {
		if (empty()) {
			return false;
		}
		if (size_ > 1) {
			data_[1] = std::move(data_[size_]);
		}
		--size_;
		if (size_ > 1) {
			sink(1);
		}
		return true;
	}

	const T& top() const noexcept {
		return data_[1];
	}

	bool empty() const noexcept {
		return size_ == 0;
	}

	bool full() const noexcept {
		return size_ == Capacity;
	}

	constexpr std::size_t size() const noexcept {
		return size_;
	}

private:
	static constexpr bool kNoThrowMove =
		std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_assignable_v<T>;
	static constexpr bool kNoThrowCopyAssign = std::is_nothrow_copy_assignable_v<T>;
	static constexpr bool kNoThrowLess = noexcept(std::declval<const T&>() < std::declval<const T&>());

	constexpr void swap_nodes(std::size_t a, std::size_t b) noexcept(kNoThrowMove) {
		// T temp = std::move(data_[a]);
		// data_[a] = std::move(data_[b]);
		// data_[b] = std::move(temp);
        std::swap(data_[a], data_[b]);
	}

	constexpr void swim(std::size_t index) noexcept(kNoThrowMove && kNoThrowLess) {
		while (index > 1) {
			const std::size_t parent = index / 2;
			if (!(data_[index] < data_[parent])) {
				break;
			}
			swap_nodes(index, parent);
			index = parent;
		}
	}

	constexpr void sink(std::size_t index) noexcept(kNoThrowMove && kNoThrowLess) {
		while (true) {
			const std::size_t left = index * 2;
			if (left > size_) {
				break;
			}
			const std::size_t right = left + 1;
			std::size_t smallest = left;
			if (right <= size_ && data_[right] < data_[left]) {
				smallest = right;
			}
			if (!(data_[smallest] < data_[index])) {
				break;
			}
			swap_nodes(index, smallest);
			index = smallest;
		}
	}

	std::array<T, Capacity + 1> data_{};
	std::size_t size_ = 0;
};

#endif