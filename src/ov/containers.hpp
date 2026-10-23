#pragma once

// Godot-HashMap-shaped flat hash tables (has / insert / erase / getptr / clear / size / is_empty /
// reserve), iterated like std's (first / second). The entries sit in one dense array (iteration is a
// linear walk, no allocation per insert); an open-addressing index (Robin Hood, linear probing,
// backward-shift erase) maps each hash to its entry. An erase moves the last entry into the hole, so:
// - any insert or erase invalidates iterators and getptr pointers into the same table;
// - iteration order is insertion order perturbed by erases, not godot's: nothing in the core may
//   depend on it (readers sort by id, counters are order-independent).

#include "ov/murmur3.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace ov {

namespace detail {

// TKeyOf::get(entry) gives the entry's key: the entry itself for a set, .first for a map.
template <typename TKey, typename TEntry, typename TKeyOf, typename THasher, typename TEqual>
class FlatTable {
    static constexpr uint32_t EMPTY = UINT32_MAX;
    static constexpr size_t MIN_SLOTS = 16;
    struct Slot {
        uint32_t hash;
        uint32_t entry; // index in m_entries, EMPTY for a free slot
    };

    std::vector<Slot> m_slots; // a power of two long, or empty
    size_t m_mask = 0;

    size_t _distance(size_t p_slot, uint32_t p_hash) const { return (p_slot - (p_hash & m_mask)) & m_mask; }

    // The slot of p_key, or EMPTY.
    size_t _find(const TKey &p_key, uint32_t p_hash) const {
        if (m_entries.empty()) {
            return EMPTY;
        }
        for (size_t i = p_hash & m_mask, dist = 0;; i = (i + 1) & m_mask, dist++) {
            const Slot &s = m_slots[i];
            // Robin Hood: past an entry nearer its home than we are from ours, the key isn't there.
            if (s.entry == EMPTY || _distance(i, s.hash) < dist) {
                return EMPTY;
            }
            if (s.hash == p_hash && TEqual()(TKeyOf::get(m_entries[s.entry]), p_key)) {
                return i;
            }
        }
    }

    // Places a slot known not to be in the index (there is room).
    void _place(Slot p_slot) {
        for (size_t i = p_slot.hash & m_mask, dist = 0;; i = (i + 1) & m_mask, dist++) {
            Slot &s = m_slots[i];
            if (s.entry == EMPTY) {
                s = p_slot;
                return;
            }
            const size_t s_dist = _distance(i, s.hash);
            if (s_dist < dist) {
                std::swap(s, p_slot);
                dist = s_dist;
            }
        }
    }

    // Index slots for p_count entries at a load factor of at most 3/4.
    void _fit(size_t p_count) {
        size_t slots = m_slots.empty() ? MIN_SLOTS : m_slots.size();
        while (p_count * 4 > slots * 3) {
            slots *= 2;
        }
        if (slots == m_slots.size()) {
            return;
        }
        const std::vector<Slot> old = std::move(m_slots);
        m_slots.assign(slots, Slot{ 0, EMPTY });
        m_mask = slots - 1;
        for (const Slot &s : old) {
            if (s.entry != EMPTY) {
                _place(s);
            }
        }
    }

protected:
    std::vector<TEntry> m_entries;

    static uint32_t _hash(const TKey &p_key) { return THasher::hash(p_key); }

    // p_key's entry (p_hash = _hash(p_key)), or nullptr.
    TEntry *_get(const TKey &p_key, uint32_t p_hash) {
        const size_t slot = _find(p_key, p_hash);
        return slot == EMPTY ? nullptr : &m_entries[m_slots[slot].entry];
    }
    const TEntry *_get(const TKey &p_key) const {
        const size_t slot = _find(p_key, _hash(p_key));
        return slot == EMPTY ? nullptr : &m_entries[m_slots[slot].entry];
    }

    // Appends p_entry, whose key (hash p_hash) isn't in the table.
    TEntry &_add(uint32_t p_hash, TEntry &&p_entry) {
        _fit(m_entries.size() + 1);
        _place(Slot{ p_hash, uint32_t(m_entries.size()) });
        m_entries.push_back(std::move(p_entry));
        return m_entries.back();
    }

    bool _erase(const TKey &p_key) {
        size_t i = _find(p_key, _hash(p_key));
        if (i == EMPTY) {
            return false;
        }
        const uint32_t hole = m_slots[i].entry;
        // Backward shift: the run after the slot moves one back, up to a free slot or an entry at home.
        for (size_t next = (i + 1) & m_mask; m_slots[next].entry != EMPTY && _distance(next, m_slots[next].hash) != 0;
                next = (next + 1) & m_mask) {
            m_slots[i] = m_slots[next];
            i = next;
        }
        m_slots[i].entry = EMPTY;
        // The last entry fills the hole: repoint its slot.
        const uint32_t last = uint32_t(m_entries.size() - 1);
        if (hole != last) {
            m_entries[hole] = std::move(m_entries[last]);
            size_t s = _hash(TKeyOf::get(m_entries[hole])) & m_mask;
            while (m_slots[s].entry != last) {
                s = (s + 1) & m_mask;
            }
            m_slots[s].entry = hole;
        }
        m_entries.pop_back();
        return true;
    }

public:
    using Iterator = typename std::vector<TEntry>::iterator;
    using ConstIterator = typename std::vector<TEntry>::const_iterator;

    Iterator begin() noexcept { return m_entries.begin(); }
    Iterator end() noexcept { return m_entries.end(); }
    ConstIterator begin() const noexcept { return m_entries.begin(); }
    ConstIterator end() const noexcept { return m_entries.end(); }

    bool is_empty() const noexcept { return m_entries.empty(); }
    size_t size() const noexcept { return m_entries.size(); }
    // Keeps the capacity, like godot's.
    void clear() noexcept {
        m_entries.clear();
        for (Slot &s : m_slots) {
            s.entry = EMPTY;
        }
    }
    // Grows only, like godot's.
    void reserve(size_t p_count) {
        _fit(p_count);
        m_entries.reserve(p_count);
    }

    bool has(const TKey &p_key) const { return _find(p_key, _hash(p_key)) != EMPTY; }
    bool erase(const TKey &p_key) { return _erase(p_key); }
};

struct SetKey {
    template <typename TKey>
    static const TKey &get(const TKey &p_entry) {
        return p_entry;
    }
};

struct MapKey {
    template <typename TKey, typename TValue>
    static const TKey &get(const std::pair<TKey, TValue> &p_entry) {
        return p_entry.first;
    }
};

} // namespace detail

// Entries are std::pair<TKey, TValue>; a caller must not write an entry's first.
template <typename TKey, typename TValue, typename THasher, typename TEqual = std::equal_to<TKey>>
class HashMap : public detail::FlatTable<TKey, std::pair<TKey, TValue>, detail::MapKey, THasher, TEqual> {
public:
    void insert(const TKey &p_key, const TValue &p_value) {
        const uint32_t hash = this->_hash(p_key);
        if (std::pair<TKey, TValue> *e = this->_get(p_key, hash)) {
            e->second = p_value;
        } else {
            this->_add(hash, std::pair<TKey, TValue>(p_key, p_value));
        }
    }
    // godot's operator[]: default-constructs a missing key's value.
    TValue &operator[](const TKey &p_key) {
        const uint32_t hash = this->_hash(p_key);
        if (std::pair<TKey, TValue> *e = this->_get(p_key, hash)) {
            return e->second;
        }
        return this->_add(hash, std::pair<TKey, TValue>(p_key, TValue())).second;
    }
    TValue *getptr(const TKey &p_key) {
        std::pair<TKey, TValue> *e = this->_get(p_key, this->_hash(p_key));
        return e ? &e->second : nullptr;
    }
    const TValue *getptr(const TKey &p_key) const {
        const std::pair<TKey, TValue> *e = this->_get(p_key);
        return e ? &e->second : nullptr;
    }
};

// The godot default: HashMapHasherDefault's spirit, over std::hash (membership-only sets in the core:
// uint32 ids). Finalized, since std::hash of an integer is the identity and the index masks low bits.
struct DefaultHash {
    template <typename T>
    static uint32_t hash(const T &p_v) {
        const uint64_t h = uint64_t(std::hash<T>{}(p_v));
        return fmix32(uint32_t(h) ^ uint32_t(h >> 32));
    }
};

template <typename TKey, typename THasher = DefaultHash, typename TEqual = std::equal_to<TKey>>
class HashSet : public detail::FlatTable<TKey, TKey, detail::SetKey, THasher, TEqual> {
public:
    void insert(const TKey &p_key) {
        const uint32_t hash = this->_hash(p_key);
        if (!this->_get(p_key, hash)) {
            this->_add(hash, TKey(p_key));
        }
    }
    // godot's remove(const Iterator &): erase at the iterator, no return value. The last entry moves
    // into its place, so removing begin() until empty pops every entry.
    void remove(const typename HashSet::Iterator &p_it) { this->_erase(TKey(*p_it)); }
};

} // namespace ov
