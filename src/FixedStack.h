#ifndef FIXED_STACK_HEADER
#define FIXED_STACK_HEADER

#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <utility>

/**
 * @brief Fixed-capacity stack implemented with std::array.
 *
 * @tparam T Element type.
 * @tparam N Maximum capacity (compile-time).
 *
 * @note This container stores elements in a fixed-size std::array<T, N>.
 *       It does not allocate dynamic memory.
 *
 * @par Example
 * @code
 * FixedStack<int, 4> s;
 * s.push(1);
 * s.emplace(2);
 * int top = s.top(); // 2
 * s.pop();
 * @endcode
 */
template <class T, std::size_t N>
class FixedStack {
	static_assert(N > 0, "FixedStack capacity N must be greater than 0");

public:
	using value_type = T;
	using size_type = std::size_t;
	using reference = T&;
	using const_reference = const T&;
	using iterator = typename std::array<T, N>::iterator;
	using const_iterator = typename std::array<T, N>::const_iterator;

	/**
	 * @brief Constructs an empty stack.
	 */
	constexpr FixedStack() noexcept(std::is_nothrow_default_constructible_v<T>) = default;

	/**
	 * @brief Returns true if the stack is empty.
	 */
	[[nodiscard]] constexpr bool empty() const noexcept {
		return size_ == 0;
	}

	/**
	 * @brief Returns true if the stack is full.
	 */
	[[nodiscard]] constexpr bool full() const noexcept {
		return size_ == N;
	}

	/**
	 * @brief Returns the number of elements currently in the stack.
	 */
	[[nodiscard]] constexpr size_type size() const noexcept {
		return size_;
	}

	/**
	 * @brief Returns the maximum capacity (N).
	 */
	[[nodiscard]] static constexpr size_type capacity() noexcept {
		return N;
	}

	/**
	 * @brief Returns a reference to the top element.
	 * @throws std::underflow_error if the stack is empty.
	 */
	reference top() {
		return data_[size_ - 1];
	}

	/**
	 * @brief Returns a const reference to the top element.
	 * @throws std::underflow_error if the stack is empty.
	 */
	const_reference top() const {
		return data_[size_ - 1];
	}

	/**
	 * @brief Pushes a copy of @p value onto the stack.
	 * @throws std::overflow_error if the stack is full.
	 */
	void push(const T& value) {
		data_[size_] = value;
		++size_;
	}

	/**
	 * @brief Pushes a moved @p value onto the stack.
	 * @throws std::overflow_error if the stack is full.
	 */
	void push(T&& value) {
		data_[size_] = std::move(value);
		++size_;
	}

	/**
	 * @brief Constructs an element in-place on top of the stack.
	 * @tparam Args Constructor argument types.
	 * @throws std::overflow_error if the stack is full.
	 * @return Reference to the newly inserted element.
	 */
	template <class... Args>
	reference emplace(Args&&... args) {
		data_[size_] = T(std::forward<Args>(args)...);
		++size_;
		return data_[size_ - 1];
	}

	/**
	 * @brief Pops the top element.
	 * @throws std::underflow_error if the stack is empty.
	 */
	void pop() {
		--size_;
	}

	/**
	 * @brief Clears the stack.
	 */
	constexpr void clear() noexcept {
		size_ = 0;
	}

	/**
	 * @brief Iterator to the bottom of the stack.
	 */
	constexpr iterator begin() noexcept {
		return data_.begin();
	}

	/**
	 * @brief Iterator past the top of the stack.
	 */
	constexpr iterator end() noexcept {
		return data_.begin() + static_cast<std::ptrdiff_t>(size_);
	}

	/**
	 * @brief Const iterator to the bottom of the stack.
	 */
	constexpr const_iterator begin() const noexcept {
		return data_.begin();
	}

	/**
	 * @brief Const iterator past the top of the stack.
	 */
	constexpr const_iterator end() const noexcept {
		return data_.begin() + static_cast<std::ptrdiff_t>(size_);
	}

	/**
	 * @brief Const iterator to the bottom of the stack.
	 */
	constexpr const_iterator cbegin() const noexcept {
		return data_.cbegin();
	}

	/**
	 * @brief Const iterator past the top of the stack.
	 */
	constexpr const_iterator cend() const noexcept {
		return data_.cbegin() + static_cast<std::ptrdiff_t>(size_);
	}

private:
	std::array<T, N> data_{};
	size_type size_ = 0;
};

#endif
