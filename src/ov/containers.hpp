#pragma once

// Godot-HashMap-shaped wrappers over the standard associative containers (has / insert / erase /
// getptr / clear / size / is_empty / reserve), iterated like std's (first / second). Iteration order
// is not godot's: nothing in the core may depend on it (readers sort by id, counters are
// order-independent).

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace ov {

// Godot's hashers expose a static hash(const TKey &), std's containers want operator(): the adapter
// bridges them so the core's godot-shaped hashers (e.g. Vector3i64Hasher) keep their shape.
template <typename TKey, typename THasher>
struct HasherAdapter {
    // noexcept: libstdc++ then doesn't cache the hash in every node.
    size_t operator()(const TKey &p_key) const noexcept { return THasher::hash(p_key); }
};

template <typename TKey, typename TValue, typename THasher, typename TEqual = std::equal_to<TKey>>
class HashMap {
    using Map = std::unordered_map<TKey, TValue, HasherAdapter<TKey, THasher>, TEqual>;
    Map m_map;

public:
    using Iterator = typename Map::iterator;
    using ConstIterator = typename Map::const_iterator;

    Iterator begin() noexcept { return m_map.begin(); }
    Iterator end() noexcept { return m_map.end(); }
    ConstIterator begin() const noexcept { return m_map.begin(); }
    ConstIterator end() const noexcept { return m_map.end(); }

    bool is_empty() const noexcept { return m_map.empty(); }
    size_t size() const noexcept { return m_map.size(); }
    void clear() noexcept { m_map.clear(); }
    // Grows only, like godot's (std's reserve rehashes to the requested size, smaller included).
    void reserve(size_t p_count) {
        if (p_count > size_t(float(m_map.bucket_count()) * m_map.max_load_factor())) {
            m_map.reserve(p_count);
        }
    }

    void insert(const TKey &p_key, const TValue &p_value) { m_map.insert_or_assign(p_key, p_value); }
    // godot's operator[]: default-constructs a missing key's value.
    TValue &operator[](const TKey &p_key) { return m_map[p_key]; }
    bool has(const TKey &p_key) const { return m_map.count(p_key) != 0; }
    TValue *getptr(const TKey &p_key) {
        const Iterator it = m_map.find(p_key);
        return it == m_map.end() ? nullptr : &it->second;
    }
    const TValue *getptr(const TKey &p_key) const {
        const ConstIterator it = m_map.find(p_key);
        return it == m_map.end() ? nullptr : &it->second;
    }
    bool erase(const TKey &p_key) { return m_map.erase(p_key) != 0; }
};

// The godot default: HashMapHasherDefault's spirit, over std::hash (membership-only sets in the core:
// uint32 ids). Iteration layout is not part of the contract.
struct DefaultHash {
    template <typename T>
    static uint32_t hash(const T &p_v) {
        return uint32_t(std::hash<T>{}(p_v));
    }
};

template <typename TKey, typename THasher = DefaultHash, typename TEqual = std::equal_to<TKey>>
class HashSet {
    using Set = std::unordered_set<TKey, HasherAdapter<TKey, THasher>, TEqual>;
    Set m_set;

public:
    using ConstIterator = typename Set::const_iterator;
    using Iterator = typename Set::iterator;

    Iterator begin() noexcept { return m_set.begin(); }
    Iterator end() noexcept { return m_set.end(); }
    ConstIterator begin() const noexcept { return m_set.begin(); }
    ConstIterator end() const noexcept { return m_set.end(); }

    bool is_empty() const noexcept { return m_set.empty(); }
    size_t size() const noexcept { return m_set.size(); }
    void clear() noexcept { m_set.clear(); }
    void reserve(size_t p_count) {
        if (p_count > size_t(float(m_set.bucket_count()) * m_set.max_load_factor())) {
            m_set.reserve(p_count);
        }
    }

    void insert(const TKey &p_key) { m_set.insert(p_key); }
    bool has(const TKey &p_key) const { return m_set.count(p_key) != 0; }
    bool erase(const TKey &p_key) { return m_set.erase(p_key) != 0; }
    // godot's remove(const Iterator &): erase at the iterator, no return value.
    void remove(const Iterator &p_it) { m_set.erase(p_it); }
};

} // namespace ov
