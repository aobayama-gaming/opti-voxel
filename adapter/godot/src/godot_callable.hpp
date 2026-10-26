#pragma once

// A godot Callable that runs a std::function: the core's posted work (RDDevice::post) and the hosts' deferred calls.

#include <godot_cpp/variant/callable.hpp>
#include <godot_cpp/variant/callable_custom.hpp>

#include <functional>
#include <utility>

class FunctionCallable : public godot::CallableCustom {
public:
    explicit FunctionCallable(std::function<void()> p_fn) : fn(std::move(p_fn)) {}
    uint32_t hash() const override { return uint32_t(uintptr_t(this)); }
    godot::String get_as_text() const override { return "FunctionCallable"; }
    CompareEqualFunc get_compare_equal_func() const override { return &equal; }
    CompareLessFunc get_compare_less_func() const override { return &less; }
    bool is_valid() const override { return true; }
    godot::ObjectID get_object() const override { return godot::ObjectID(); }
    void call(const godot::Variant **, int, godot::Variant &, GDExtensionCallError &r_call_error) const override {
        fn();
        r_call_error.error = GDEXTENSION_CALL_OK;
    }

private:
    std::function<void()> fn;
    static bool equal(const CallableCustom *p_a, const CallableCustom *p_b) { return p_a == p_b; }
    static bool less(const CallableCustom *p_a, const CallableCustom *p_b) { return p_a < p_b; }
};

inline godot::Callable godot_callable(std::function<void()> p_fn) {
    return godot::Callable(memnew(FunctionCallable(std::move(p_fn))));
}
