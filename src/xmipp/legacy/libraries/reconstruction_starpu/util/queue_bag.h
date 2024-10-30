#ifndef XMIPP_QUEUE_BAG_H
#define XMIPP_QUEUE_BAG_H

#include <vector>
#include <type_traits>

#define RFS_QUEUE_BAG_RUN_TESTS 0

namespace rfs {

	void testQueueBag();

	/**
	 * A circular queue with relaxed ordering requirements.
	 * Some operations may change the order of queued elements for performance.
	 * This is always mentioned in the documentation.
	 *
	 * @tparam T any trivially copyable type
	 */
	template<typename T>
	class queue_bag {
		static_assert(std::is_trivially_copyable<T>::value,
		              "queue_bag is designed only for trivially copyable elements");

	private:
		/** Contains the values in the queue. Head and tail indices go in a circle around this array, wrapping at the end. */
		T *elements = nullptr;
		size_t elementsCapacity = 0;

		/** Index of first element. Logically smaller than tail. Unless empty, it points to a valid element inside queue. */
		size_t head = 0;
		/** Index of last element. Logically bigger than head. Usually points to an empty position, but points to the head when full
		 * (size == values.length). */
		size_t tail = 0;
		/** Number of elements in the queue. */
		size_t size = 0;

		size_t internalIndex(size_t externalIndex) const {
			assert(size > 0);
			assert(externalIndex < size);
			size_t index = head + externalIndex;
			if (index >= elementsCapacity) {
				index -= elementsCapacity;
			}
			return index;
		}

	public:

#if RFS_QUEUE_BAG_RUN_TESTS
		queue_bag() {
			testQueueBag();
		}
#endif

		/** Append given object to the tail. (enqueue to tail) Unless backing array needs resizing, operates in O(1) time.
		 * @param object can be null */
		void addLast(T object) noexcept {
			if (size == elementsCapacity) {
				resize(elementsCapacity * 2);
			}

			elements[tail++] = object;
			if (tail == elementsCapacity) {
				tail = 0;
			}
			size++;
		}

		/** Prepend given object to the head. (enqueue to head) Unless backing array needs resizing, operates in O(1) time.
		 * @see #addLast(Object)
		 * @param object can be null */
		void addFirst(T object) noexcept {
			if (size == elementsCapacity) {
				resize(elementsCapacity * 2);
			}

			if (head == 0) {
				head = elementsCapacity - 1;
			} else {
				head--;
			}
			elements[head] = object;
			size++;
		}

		/** Increases the size of the backing array to accommodate the specified number of additional items.
		 * Useful before adding many items to avoid multiple backing array resizes. */
		void ensureCapacity(size_t additional) noexcept {
			size_t needed = size + additional;
			if (elementsCapacity < needed) {
				resize(needed);
			}
		}

		/** Resize backing array. newSize must be bigger than current size. */
		void resize(int newSize) noexcept {
			if (newSize < 32) {
				newSize = 32;
			}
			assert(newSize > elementsCapacity);

			T *newArray = (T*)malloc(sizeof(T) * newSize);
			if (head < tail) {
				// Continuous
				memcpy(newArray, elements + head, sizeof(T) * size);
			} else if (size > 0) {
				// Wrapped
				size_t rest = elementsCapacity - head;
				assert(rest + tail == size);
				memcpy(newArray, elements + head, sizeof(T) * rest);
				memcpy(newArray + rest, elements, sizeof(T) * tail);
			}

			free(elements);
			elements = newArray;
			elementsCapacity = newSize;
			head = 0;
			tail = size;
		}

		/** Remove the first item from the queue. Always O(1).
		 * Undefined behavior when empty.
		 * @return removed object */
		T removeFirst() noexcept {
			assert(size > 0);

			size_t index = head;
			head++;
			if (head == elementsCapacity) {
				head = 0;
			}
			size--;

			return elements[index];
		}

		/** Remove the last item from the queue. Always O(1).
		 * Undefined behavior when empty.
		 * @return removed object */
		T removeLast() noexcept {
			assert(size > 0);

			if (tail == 0) {
				tail = elementsCapacity - 1;
			} else {
				tail--;
			}
			size--;
			return elements[tail];
		}

		/** Removes and returns the item at the specified index.
		 * Does not maintain any order - unless the element is first or last, the last element will take its place,
		 * eliminating copies.
		 * Undefined behavior when index is out of valid range [0, size). */
		T removeIndex(int index) noexcept {
			assert(index >= 0);
			assert(index < size);

			if (index == 0) {
				return removeFirst();
			} else if (index + 1 == size) {
				return removeLast();
			} else {
				// Remove last, put it in place of this one
				size_t elementsIndex = internalIndex(index);
				T result = elements[elementsIndex];
				elements[elementsIndex] = removeLast();
				return result;
			}
		}

		size_t getSize() const noexcept {
			return size;
		}

		/** Returns true if the queue is empty. */
		bool empty() const noexcept {
			return size == 0;
		}

		/** Returns the first (head) item in the queue (without removing it).
		 * Undefined behavior when empty. */
		const T &first() const noexcept {
			return elements[internalIndex(0)];
		}

		T &first() noexcept {
			return elements[internalIndex(0)];
		}

		/** Returns the last (tail) item in the queue (without removing it).
		 * Undefined behavior when empty. */
		const T &last() const noexcept {
			return elements[internalIndex(size - 1)];
		}

		T &last() noexcept {
			return elements[internalIndex(size - 1)];
		}

		/** Retrieves the value in queue without removing it. Indexing is from the front to back, zero based. Therefore get(0) is the
		 * same as {@link #first()}.
		 * Undefined behavior when index is out of valid range [0, size). */
		const T &operator[](int index) const noexcept {
			return elements[internalIndex(index)];
		}

		T &operator[](int index) noexcept {
			return elements[internalIndex(index)];
		}

		~queue_bag() {
			free(elements);
		}
	};

	/** Iterate through queue_bag until all elements are consumed or until consumer returns false. */
	template<typename T, typename Consumer>
	void forEach(queue_bag<T>& queue, Consumer consumer) {
		for (size_t i = 0; i < queue.getSize(); ++i) {
			if (!consumer(queue[i])) {
				return;
			}
		}
	}

	enum class RemoveResult {
		KeepContinue,
		KeepBreak,
		RemoveContinue,
		RemoveBreak
	};

	/**
	 * Does bulk removal of elements from queue_bag. Removes only those elements, which match the predicate.
	 * @tparam Predicate lambda that takes T& and returns RemoveResult
	 * @return amount of removed items
	 */
	template<typename T, typename Predicate>
	size_t removeMatching(queue_bag<T>& queue, Predicate predicate) {
		size_t removed = 0;
		size_t i = 0;
		while (i < queue.getSize()) {
			RemoveResult res = predicate(queue[i]);
			if (res == RemoveResult::RemoveContinue || res == RemoveResult::RemoveBreak) {
				queue.removeIndex(i);
			} else {
				i++;
			}

			if (res == RemoveResult::KeepBreak || res == RemoveResult::RemoveBreak) {
				break;
			}
		}
		return removed;
	}

#if RFS_QUEUE_BAG_RUN_TESTS
	inline void testQueueBag() {
		static bool tested = false;
		if (tested) {
			return;
		}
		tested = true;

		queue_bag<int> queue;
		queue.addFirst(42);
		int stairs = 20;

		for (int i = 0; i < stairs; ++i) {
			queue.addFirst(i);
			queue.addLast(i);
			assert(queue.first() == i);
			assert(queue.last() == i);
			assert(queue[queue.getSize() / 2] == 42);
		}

		for (int i = 0; i < stairs; ++i) {
			assert(queue[i] == stairs - 1 - i);
			assert(queue[stairs + 1 + i] == i);
		}

		int stairCheckIndex = 0;
		forEach(queue, [&stairCheckIndex, stairs](const int value){
			if (stairCheckIndex < stairs) {
				assert(value == stairs - 1 - stairCheckIndex);
			} else if (stairCheckIndex == stairs) {
				assert(value == 42);
			} else {
				assert(stairCheckIndex - stairs - 1 == value);
			}
			stairCheckIndex++;
			return true;
		});

		for (int i = stairs-1; i >= 0; --i) {
			assert(queue.removeFirst() == i);
			assert(queue.removeLast() == i);
		}
		assert(queue.removeFirst() == 42);
	}
#endif
}

#endif //XMIPP_QUEUE_BAG_H
