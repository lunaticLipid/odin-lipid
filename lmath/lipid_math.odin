package lm

import "core:os"
using import "core:math"

_zerof32: f32 = 0;
_zerof64: f64 = 0;
inf32 := f32(1) / _zerof32;
inf64 := f64(1) / _zerof64;

when os.OS == "linux" {
    foreign import libc "system:c"
    foreign libc {
        @(link_name="atan2f")   atan2_f32   :: proc(y, x: f32) -> f32 ---;
        @(link_name="atan2")    atan2_f64   :: proc(y, x: f64) -> f64 ---;
    }
}
when os.OS == "windows" {
    //https://www.dsprelated.com/showarticle/1052.php
    atan2_f64 :: proc(y, x: f64) -> f64 {
        ay, ax := abs(y), abs(x);
        invert := ay > ax;
        z := invert ? ax / ay : ay / ax;
        th := (0.97239411 + -0.19194795 * z * z) * z;
        if invert do th = PI / 2 - th;
        if x < 0 do th = PI - th;
        if y < 0 do th = -abs(th);
        else do th = abs(th);
        return th;
    }

    atan2_f32 :: proc(y, x: f32) -> f32 {
        return cast(f32) atan2_f64(f64(y), f64(x));
    }
}

atan2 :: proc{atan2_f32, atan2_f64};

damp :: proc(a, b: $T, lambda, dt: f32) -> T {
    return lerp(a, b, 1 - pow(E, -lambda * dt));
}

wrap :: proc(n, m: int) -> int do return (n % m + m) % m;
// clamp :: proc(x, lower, upper: $T) -> T do return min(max(x, lower), upper);
nonDescending :: proc(a, b, c: $T) -> bool do return a <= b && b <= c;
ascending :: proc(a, b, c: $T) -> bool do return a < b && b < c;

ceilSqrt :: proc(n: int) -> int {
    if n == 0 do return n;

    i := 1; result := 1;
    for result < n {
        i += 1;
        result = i*i;
    }

    return i;
}

perp :: proc(v: Vec2) -> Vec2 do return Vec2{-v.y, v.x};
perpdot :: proc{cross2};

distance :: proc(a, b: Vec2) -> f32 do return length(a - b);

aff :: proc{affVec2, affVec3};
aff0 :: proc{aff0Vec2, aff0Vec3};

affVec2 :: proc(v: Vec2, m: Mat3) -> Vec2 {
    affine_result := vec3_mul_mat3(m, Vec3{v.x, v.y, 1});
    return Vec2{affine_result.x, affine_result.y} / affine_result.z;
}

aff0Vec2 :: proc(v: Vec2, m: Mat3) -> Vec2 {
    affine_result := vec3_mul_mat3(m, Vec3{v.x, v.y, 0});
    return Vec2{affine_result.x, affine_result.y};
}

affVec3 :: proc(v: Vec3, m: Mat4) -> Vec3 {
    affine_result := vec4_mul_mat4(m, Vec4{v.x, v.y, v.z, 1});
    return Vec3{affine_result.x, affine_result.y, affine_result.z} / affine_result.w;
}

aff0Vec3 :: proc(v: Vec3, m: Mat4) -> Vec3 {
    affine_result := vec4_mul_mat4(m, Vec4{v.x, v.y, v.z, 0});
    return Vec3{affine_result.x, affine_result.y, affine_result.z};
}

mat3_mul_many :: proc(ms: ..Mat3) -> Mat3 {
    c := identity(Mat3);
    for m in ms do c = mul(c, m);
	return c;
}

mat4_mul_many :: proc(ms: ..Mat4) -> Mat4 {
    c := identity(Mat4);
    for m in ms do c = mul(c, m);
	return c;
}

mul_many :: proc{mat3_mul_many, mat4_mul_many};

mat2_mul_vec2 :: proc(m: Mat2, v: Vec2) -> Vec2 {
    return Vec2{
        m[0][0] * v[0] + m[1][0] * v[1],
        m[0][1] * v[0] + m[1][1] * v[1]
    };
}

vec2_mul_mat2 :: proc(v: Vec2, m: Mat2) -> Vec2 {
	return Vec2{
		v[0]*m[0][0] + v[1]*m[0][1],
		v[0]*m[1][0] + v[1]*m[1][1],
	};
}

vec3_mul_mat3 :: proc(m: Mat3, v: Vec3) -> Vec3 {
	return Vec3{
		v[0]*m[0][0] + v[1]*m[0][1] + v[2]*m[0][2],
		v[0]*m[1][0] + v[1]*m[1][1] + v[2]*m[1][2],
		v[0]*m[2][0] + v[1]*m[2][1] + v[2]*m[2][2],
	};
}

vec4_mul_mat4 :: proc(m: Mat4, v: Vec4) -> Vec4 {
	return Vec4{
		v[0]*m[0][0] + v[1]*m[0][1] + v[2]*m[0][2] + v[3]*m[0][3],
		v[0]*m[1][0] + v[1]*m[1][1] + v[2]*m[1][2] + v[3]*m[1][3],
		v[0]*m[2][0] + v[1]*m[2][1] + v[2]*m[2][2] + v[3]*m[2][3],
		v[0]*m[3][0] + v[1]*m[3][1] + v[2]*m[3][2] + v[3]*m[3][3],
	};
}

mul_post :: proc{vec2_mul_mat2, vec3_mul_mat3, vec4_mul_mat4};

mat2_rotate_post :: proc(angle_radians: f32) -> Mat2 {
	c := cos(angle_radians);
	s := sin(angle_radians);
	return Mat2{{c, -s}, {s, c}};
}

mat2_clockwise_post :: proc(angle_radians: f32) -> Mat2 {
	c := cos(angle_radians);
	s := sin(angle_radians);
	return Mat2{{c, s}, {-s, c}};
}

mat3_translate_post_aff :: proc(v: Vec2) -> Mat3 {
	m := identity(Mat3);
	m[0][2] = v[0];
	m[1][2] = v[1];
	return m;
}

mat3_scale_aff :: proc(s: Vec2) -> Mat3 {
    m: Mat3;
    m[0][0] = s.x;
    m[1][1] = s.y;
    m[2][2] = 1;
    return m;
}

// TODO: Doesn't work now, but it used to:
// mat3_scale_post_aff :: mat3_scale_aff;
// ...instead, COPIED☠️ :
mat3_scale_post_aff :: proc(s: Vec2) -> Mat3 {
    m: Mat3;
    m[0][0] = s.x;
    m[1][1] = s.y;
    m[2][2] = 1;
    return m;
}

mat3_clockwise_post_aff :: proc(angle_radians: f32) -> (result: Mat3) {
    result = identity(Mat3);

	c := cos(angle_radians);
	s := sin(angle_radians);

    result[0][0] = c;
    result[0][1] = s;
    result[1][0] = -s;
    result[1][1] = c;

	return;
}

// Computes the inverse treating the matrix as row-major as opposed to other functions in this file,
// but that's fine, because that means it computes the inverse of transpose of m and transposes it
// again. inv(trans(A)) == trans(inv(A))
mat3_inverse :: proc(m: Mat3) -> Mat3 {
    o: Mat3;

    det := m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
    m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
    m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    invdet := 1 / det;

    o[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * invdet;
    o[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * invdet;
    o[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * invdet;
    o[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * invdet;
    o[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * invdet;
    o[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * invdet;
    o[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * invdet;
    o[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * invdet;
    o[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * invdet;

    return o;
}

mat4_rotate_post_aff :: proc(v: Vec3, angle_radians: f32) -> Mat4 {
	c := cos(angle_radians);
	s := sin(angle_radians);

	a := norm(v);
	t := a * (1-c);

	rot := identity(Mat4);

	rot[0][0] = c + t[0]*a[0];
	rot[1][0] = 0 + t[0]*a[1] + s*a[2];
	rot[2][0] = 0 + t[0]*a[2] - s*a[1];

	rot[0][1] = 0 + t[1]*a[0] - s*a[2];
	rot[1][1] = c + t[1]*a[1];
	rot[2][1] = 0 + t[1]*a[2] + s*a[0];

	rot[0][2] = 0 + t[2]*a[0] + s*a[1];
	rot[1][2] = 0 + t[2]*a[1] - s*a[0];
	rot[2][2] = c + t[2]*a[2];

	return rot;
}

pointInAABB :: proc(point, bottom_left, size: Vec2) -> bool {
    return point.x >= bottom_left.x && point.x <= bottom_left.x + size.x &&
    point.y >= bottom_left.y && point.y <= bottom_left.y + size.y;
}

pointInUnorderedAABB :: proc(boxA, boxB, p: Vec2) -> bool {
    return (
        p.x <= max(boxA.x, boxB.x) && p.x >= min(boxA.x, boxB.x) &&
        p.y <= max(boxA.y, boxB.y) && p.y >= min(boxA.y, boxB.y)
    );
}

pointInUnorderedAABBWeak :: proc(boxA, boxB, p: Vec2) -> bool {
    return (
        p.x < max(boxA.x, boxB.x) && p.x > min(boxA.x, boxB.x) &&
        p.y < max(boxA.y, boxB.y) && p.y > min(boxA.y, boxB.y)
    );
}

Winding :: enum { Clockwise, CounterClockwise, Degenerate };

triangleWinding :: proc(a, b, c: Vec2) -> Winding {
    pd := perpdot(b - a, c - a);
    if pd == 0 do return Winding.Degenerate;
    else if pd > 0 do return Winding.CounterClockwise;
    else do return Winding.Clockwise;
}

polygonWinding :: proc(p: []Vec2) -> Winding {
    assert(len(p) > 2);
    low_right_i: int;
    low_right := p[0];

    for i in 1 .. len(p)-1 {
        if (
            p[i].y < low_right.y ||
            (p[i].y == low_right.y && p[i].x > low_right.x)
        ) {
            low_right_i = i;
            low_right = p[i];
        }
    }

    previous_i := wrap(low_right_i - 1, len(p));
    next_i := wrap(low_right_i + 1, len(p));
    return triangleWinding(p[previous_i], p[low_right_i], p[next_i]);
}

// Check if vector b points between a and c in clockwise order.
// True if b is aligned with a or c.
liesBetween :: proc(a, b, c: Vec2) -> bool {
    assert(a != Vec2{0, 0});
    assert(b != Vec2{0, 0});
    assert(c != Vec2{0, 0});
    if a == b || c == b do return true;

    pd_a, pd_c := perpdot(a, b), perpdot(c, b);

    if perpdot(c, a) >= 0 do return (pd_a <= 0 && pd_c >= 0);
    else do return (pd_a <= 0 || pd_c >= 0);
}

lineSegmentsIntersect :: proc(a0, a1, b0, b1: Vec2) -> bool {
    using Winding;

    w0 := triangleWinding(a0, a1, b0);
    w1 := triangleWinding(a0, a1, b1);
    w2 := triangleWinding(b0, b1, a0);
    w3 := triangleWinding(b0, b1, a1);

    if w0 != w1 && w2 != w3 do return true;

    if (w0 == Degenerate && pointInUnorderedAABB(a0, a1, b0)) ||
    (w1 == Degenerate && pointInUnorderedAABB(a0, a1, b1)) ||
    (w2 == Degenerate && pointInUnorderedAABB(b0, b1, a0)) ||
    (w3 == Degenerate && pointInUnorderedAABB(b0, b1, a1)) {
        return true;
    }

    return false;
}

// Weak test: endpoints aren't considered a part of the segment.
lineSegmentsIntersectWeak :: proc(a0, a1, b0, b1: Vec2) -> bool {
    if (a0 == b0 && a1 == b1) || (a0 == b1 && a1 == b0) do return true;
    using Winding;

    w0 := triangleWinding(a0, a1, b0);
    w1 := triangleWinding(a0, a1, b1);
    w2 := triangleWinding(b0, b1, a0);
    w3 := triangleWinding(b0, b1, a1);

    if w0 == Degenerate || w1 == Degenerate || w2 == Degenerate || w3 == Degenerate {
        if w0 == Degenerate && w1 == Degenerate {
            if (
                pointInUnorderedAABBWeak(a0, a1, b0) ||
                pointInUnorderedAABBWeak(a0, a1, b1) ||
                pointInUnorderedAABBWeak(b0, b1, a0)
            ) {
                return true;
            }
        }

        return false;
    }

    return w0 != w1 && w2 != w3;
}

circleIntersections :: proc(p0: Vec2, r0: f32, p1: Vec2, r1: f32) -> ([2]Vec2, int) {
    dist := distance(p0, p1);

    if dist > r0 + r1 || dist < abs(r0 - r1) do return [2]Vec2{}, 0;

    // Distance to the crossing of the line containing circle centers and
    // the line containing circle intersections.
    dist_to_crossing := ((r0*r0) - (r1*r1) + (dist*dist)) / (2 * dist);
    crossing := p0 + (p1 - p0) * (dist_to_crossing / dist);

    if dist_to_crossing == r0 do return [2]Vec2{crossing, Vec2{}}, 1;

    dist_crossing_to_intersection := sqrt(r0 * r0 - dist_to_crossing * dist_to_crossing);

    crossing_to_intersection := perp(p1 - p0) * dist_crossing_to_intersection / dist;

    return [2]Vec2{
        crossing + crossing_to_intersection,
        crossing - crossing_to_intersection
    }, 2;
}

circleLineSegmentIntersections :: proc(circle_p: Vec2, radius: f32, a, b: Vec2) -> ([2]Vec2, int) {
    dist := distanceToLine(circle_p, a, b);
    segment_len := distance(a, b);
    segment_vec := norm(b - a);

    if dist > radius do return [2]Vec2{}, 0;

    circle_projection, circle_t := projectPointOnLine(circle_p, a, b);

    if dist == radius {
        if nonDescending(f32(0), circle_t, f32(1)) {
            return [2]Vec2{circle_projection, Vec2{}}, 1;
        }
        else do return [2]Vec2{}, 0;
    }

    dist_proj_to_intersection := sqrt((radius * radius) - (dist * dist));

    int0_t := circle_t + dist_proj_to_intersection / segment_len;
    int1_t := circle_t - dist_proj_to_intersection / segment_len;

    if nonDescending(f32(0), int0_t, f32(1)) {
        int0 := lerp(a, b, int0_t);

        if nonDescending(f32(0), int1_t, f32(1)) {
            int1 := lerp(a, b, int1_t);
            return [2]Vec2{int0, int1}, 2;
        }
        else {
            return [2]Vec2{int0, Vec2{}}, 1;
        }
    }
    else if nonDescending(f32(0), int1_t, f32(1)) {
        int1 := lerp(a, b, int1_t);
        return [2]Vec2{int1, Vec2{}}, 1;
    }
    else {
        return [2]Vec2{}, 0;
    }
}

// Returns projection point and theta such that lerp(a, b, t) == projection.
projectPointOnLine :: proc(p, a, b: Vec2) -> (Vec2, f32) {
    a_to_b := b - a;
    dist := distance(a, b);
    t := dot(a_to_b, p - a) / (dist * dist);
    return a + a_to_b * t, t;
}

lineSegmentIntersection :: proc(a0, a1, b0, b1: Vec2) -> (Vec2, bool) {
    vec0 := a1 - a0;
    vec1 := b1 - b0;
    if cross(vec0, vec1) == 0 do return Vec2{}, false;

    t0 := cross(a0 - b0, vec1) / cross(vec1, vec0);
    t1 := cross(b0 - a0, vec0) / cross(vec0, vec1);

    if nonDescending(f32(0), t0, f32(1)) && nonDescending(f32(0), t1, f32(1)) {
        return a0 + vec0 * t0, true;
    }
    else do return Vec2{}, false;
}

// COPIED☠️  from lineSegmentIntersection.
lineSegmentIntersectionWeak :: proc(a0, a1, b0, b1: Vec2) -> (Vec2, bool) {
    vec0 := a1 - a0;
    vec1 := b1 - b0;
    if cross(vec0, vec1) == 0 do return Vec2{}, false;

    t0 := cross(a0 - b0, vec1) / cross(vec1, vec0);
    t1 := cross(b0 - a0, vec0) / cross(vec0, vec1);

    // Difference from copy: nonDescending -> ascending.
    if ascending(f32(0), t0, f32(1)) && ascending(f32(0), t1, f32(1)) {
        return a0 + vec0 * t0, true;
    }
    else do return Vec2{}, false;
}

// Area of simple polygon if its vertices are listed in counterclockwise order. Negative area otherwise.
signedArea :: proc(polygon: []Vec2) -> f32 {
    double_area: f32 = 0.0;
    for _, i in polygon {
        j := (i + 1) % len(polygon);
        a, b := polygon[i], polygon[j];
        double_area += (a.x - b.x) * (a.y + b.y);
    }
    return double_area / 2;
}

distanceToLine :: proc(point, a, b: Vec2) -> f32{
    line_normal := perp(norm(-b+a));
    return abs(dot(point - a, line_normal));
}

angleFromVec :: proc(v0, v1: Vec2) -> f32 {
    x := dot(v0, v1);
    y := cross(v0, v1);
    return atan2(y, x);
}

angleFromXAxis :: proc(v: Vec2) -> f32 {
    return atan2(v.y, v.x);
}
