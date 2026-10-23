#pragma once

// Godot-HashMap-shaped wrappers over the standard associative containers, so the core's call sites
// keep their shape without godot_cpp. The hash order (godot's HashMap growth = primes, std's power
// of two) differs: nothing in the core may depend on iteration order — readers sort by id, counters
// are order-independent (tools/check_core_purity.sh keeps godot includes out).
//
// Note for liboptivoxel: keep these methods to the godot call surface the core already uses (has /
// insert / erase / getptr / clear / size / is_empty / reserve), not a general container library.

#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>

namespace ov {

// Godot's hashers expose a static hash(const TKey &), std's containers want operator(): the adapter
// bridges them so the core's godot-shaped hashers (e.g. Vector3i64Hasher) keep their shape.
template <typename TKey, typename THasher>
struct HasherAdapter {
    size_t operator()(const TKey &p_key) const { return THasher::hash(p_key); }
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
    void reserve(size_t p_count) { m_map.reserve(p_count); }

    void insert(const TKey &p_key, const TValue &p_value) { m_map.insert_or_assign(p_key, p_value); }
    bool has(const TKey &p_key) const { return m_map.find(p_key) != m_map.end(); }
    TValue *getptr(const TKey &p_key) { return m_map.count(p_key) ? &m_map.find(p_key)->second : nullptr; }
    const TValue *getptr(const TKey &p_key) const { return m_map.count(p_key) ? &m_map.find(p_key)->second : nullptr; }
    bool erase(const TKey &p_key) { return m_map.erase(p_key) != 0; }
};

template <typename TKey, typename THasher, typename TEqual = std::equal_to<TKey>>
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
    void reserve(size_t p_count) { m_set.reserve(p_count); }

    void insert(const TKey &p_key) { m_set.insert(p_key); }
    bool has(const TKey &p_key) const { return m_set.count(p_key) != 0; }
    bool erase(const TKey &p_key) { return m_set.erase(p_key) != 0; }
};

} // namespace ov
