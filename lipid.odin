package lipid

import "core:fmt"
import "core:strings"

intToRawptr :: proc(offset: int) -> rawptr {
    return rawptr(uintptr(offset));
}

/* Printing */

_print_limited_counter := 10;
printLimited :: proc(args: ..any) {
    if _print_limited_counter > 0 {
        _print_limited_counter -= 1;
        fmt.println(..args);
    }
}

/* Arrays and Slices */

resetDynamic :: proc(array: ^$D/[dynamic]$T) {
    array^ = make(D);
}

sizeOfSlice :: proc(s: []$T) -> int do return size_of(T) * len(s);

reverseSlice :: proc(s: []$T) {
    for i in 0..(len(s) / 2)-1 {
        j := len(s) - 1 - i;
        s[i], s[j] = s[j], s[i];
    }
}

import "core:runtime"

unorderedRemove :: proc(array: ^[dynamic]$T, index: int, loc := #caller_location) {
    runtime.bounds_check_error_loc(loc, index, len(array));
    array[index] = array[len(array)-1];
    pop(array);
}

orderedRemove :: proc(array: ^[dynamic]$T, index: int, loc := #caller_location) {
    runtime.bounds_check_error_loc(loc, index, len(array));
    copy(array[index:], array[index+1:]);
    pop(array);
}

/* String operations */

isASCIIWhitespace :: proc(c: byte) -> bool {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

countRunes :: proc(s: string) -> (count: int = 0) {
    for r in s do count += 1;
    return;
}

split :: proc(s: string, splitter: byte = 0) -> []string {
    words_dyn: [dynamic]string;

    isSplitter :: proc(b, s: byte) -> bool {
        if s == 0 do return isASCIIWhitespace(b);
        else do return b == s;
    }

    start := 0;
    for {
        for start < len(s) {
            if isSplitter(s[start], splitter) do start += 1;
            else do break;
        }

        end := start;

        for end < len(s) {
            if !isSplitter(s[end], splitter) do end += 1;
            else do break;
        }

        if start >= len(s) do break;

        append(&words_dyn, s[start:end]);

        start = end;
    }

    return words_dyn[:];
}

getLines :: proc(str: string) -> []string {
    result: [dynamic]string;

    current_line_start := 0;
    for i := 0; i <= len(str); i += 1 {
        if i == len(str) || str[i] == '\n' {
            line_end := i;
            if i > 0 && str[i - 1] == '\r' do line_end = i - 1;
            append(&result, str[current_line_start:line_end]);
            current_line_start = i + 1;
        }
    }

    return result[:];
}

// Returns inclusive:exclusive index pairs.
getLineIndices :: proc(str: string) -> [][2]int {
    result: [dynamic][2]int;

    current_line_start := 0;
    for i := 0; i <= len(str); i += 1 {
        if i == len(str) || str[i] == '\n' {
            line_end := i;
            if i > 0 && str[i - 1] == '\r' do line_end = i - 1;
            append(&result, [2]int{current_line_start, line_end});
            current_line_start = i + 1;
        }
    }

    return result[:];
}

// Returns inclusive:exclusive index pairs.
getTrimmedNonBlankLineIndices :: proc(str: string) -> [][2]int {
    result: [dynamic][2]int;

    found_line_start := false;
    current_line_start := -1;
    for i := 0; i < len(str); i += 1 {
        if !found_line_start && !isASCIIWhitespace(str[i]) {
            found_line_start = true;
            current_line_start = i;
        }

        if str[i] == '\n' {
            if found_line_start {
                line_end := i;
                for isASCIIWhitespace(str[line_end]) do line_end -= 1;
                append(&result, [2]int{current_line_start, line_end + 1});
            }
            found_line_start = false;
        }
    }

    return result[:];
}

// Remove initial newline and indentation from multiline literal
// Indent has to be just spaces or just tabs.
// Doesn't work with \r\n.
cleanMultilineLiteral :: proc(str: string) -> string {
    if len(str) <= 1 do return str;

    start := str[0] == '\n' ? 1 : 0;
    s := cast([]byte) str[start:];

    Line :: struct {
        start, end: int,
        is_empty: bool,
    }
    lines: [dynamic]Line; defer delete(lines);

    line_start := 0;
    min_indent_length := 99999; // TODO: Actual max int.
    current_indent_length := 0;
    found_first_character_in_line := false;

    for i := 0; i < len(s); i += 1 {
        if !found_first_character_in_line {
            if s[i] == ' ' || s[i] == '\t' {
                current_indent_length += 1;
            } else if s[i] != '\n' {
                found_first_character_in_line = true;
                if current_indent_length < min_indent_length {
                    min_indent_length = current_indent_length;
                }
            }
        }

        if s[i] == '\n' {
            append(&lines, Line{line_start, i, !found_first_character_in_line});

            line_start = i + 1;
            found_first_character_in_line = false;
            current_indent_length = 0;
        }
    }

    result: strings.Builder;

    for line in lines {
        if line.is_empty {
            strings.write_string(&result, "\n");
        } else {
            strings.write_bytes(&result, s[line.start + min_indent_length : line.end + 1]);
        }
    }

    return strings.to_string(result);
}

/* Collections */

LinkedListNode :: struct(T: typeid) {
    value: T, next: ^LinkedListNode(T)
}

LinkedList :: struct(T: typeid) {
    head: ^LinkedListNode(T),
    length: int,
}

LL_print :: proc(l: LinkedList($T)) {
    n := l.head;
    for n != nil {
        if n.next == nil do print(n.value);
        else do printf("%v, ", n.value);
        n = n.next;
    }
}

toLinkedList :: proc(s: ..$T) -> (result: LinkedList(T)) {
    result.length = len(s);
    current := &result.head;

    for e in s {
        current^ = new(LinkedListNode(T));
        current^.value = e;
        current = &current^.next;
    }

    return;
}

LL_reverse :: proc(l: ^LinkedList($T)) {
    current := l.head;
    new_head, previous: ^LinkedListNode(T);

    for current != nil {
        next := current.next;

        new_head = current;
        new_head.next = previous;

        previous = new_head;
        current = next;
    }

    l.head = new_head;
}

LL_push :: proc(l: ^LinkedList($T), v: T) {
    l.head = new_clone(LinkedListNode(T){v, l.head});
    l.length += 1;
}

LL_pop :: proc(l: ^LinkedList($T)) -> T {
    assert(l.head != nil, "Pop on empty LinkedList.");

    old_head := l.head;
    old_value := old_head.value;
    l.head = old_head.next;
    free(old_head);

    l.length -= 1;

    return old_value;
}

LL_insert :: proc(l: ^LinkedList($T), index: int, v: T) {
    assert(index >= 0 && index <= l.length, "LinkedList insert out of bounds.");

    address_of_previous_link := &l.head;
    current := l.head;

    for i := 0; i != index; i += 1 {
        address_of_previous_link = &current.next;
        current = current.next;
    }

    address_of_previous_link^ = new_clone(LinkedListNode(T){v, current});
    l.length += 1;
}

LL_deleteIndex :: proc(l: ^LinkedList($T), index: int) {
    assert(index >= 0 && index < l.length, "LinkedList deleteIndex out of bounds.");

    address_of_previous_link := &l.head;
    current := l.head;

    for i := 0; i != index; i += 1 {
        address_of_previous_link = &current.next;
        current = current.next;
    }

    address_of_previous_link^ = current.next;
    free(current);
    l.length -= 1;
}

DoublyLinkedListNode :: struct(T: typeid) {
    value: T, previous, next: ^DoublyLinkedListNode(T)
}

DoublyLinkedList :: struct(T: typeid) {
    head: ^DoublyLinkedListNode(T),
    length: int,
}

toDoublyLinkedList :: proc(s: ..$T) -> (result: DoublyLinkedList(T)) {
    result.length = len(s);
    address_of_previous_link := &result.head;
    previous: ^DoublyLinkedListNode(T) = nil;

    for e in s {
        address_of_previous_link^ = new(DoublyLinkedListNode(T));
        current := address_of_previous_link^;
        current.value = e;
        current.previous = previous;
        previous = current;
        address_of_previous_link = &current.next;
    }
    
    return;
}

// Entirely COPIED☠️  from LL_print
DLL_print :: proc(l: DoublyLinkedList($T)) {
    n := l.head;
    for n != nil {
        if n.next == nil do print(n.value);
        else do printf("%v, ", n.value);
        n = n.next;
    }
}

// Mostly COPIED☠️  from LL_insert
DLL_insert :: proc(l: ^DoublyLinkedList($T), index: int, v: T) {
    assert(index >= 0 && index <= l.length, "LinkedList insert out of bounds.");

    address_of_previous_link := &l.head;
    current := l.head;
    previous: ^DoublyLinkedListNode(T) = nil;

    for i := 0; i != index; i += 1 {
        address_of_previous_link = &current.next;
        previous = current;
        current = current.next;
    }

    new_node := new_clone(DoublyLinkedListNode(T){v, previous, current});
    if new_node.next != nil do new_node.next.previous = new_node;
    address_of_previous_link^ = new_node;
    l.length += 1;
}

CircularDoublyLinkedList :: struct(T: typeid) {
    head: ^DoublyLinkedListNode(T),
    length: int,
}

toCircularDoublyLinkedList :: proc(s: ..$T) -> (result: CircularDoublyLinkedList(T)) {
    if s == nil do return;

    result.length = len(s);
    address_of_previous_link := &result.head;
    previous, current: ^DoublyLinkedListNode(T) = nil, nil;

    for e in s {
        address_of_previous_link^ = new(DoublyLinkedListNode(T));
        current = address_of_previous_link^;
        current.value = e;
        current.previous = previous;
        previous = current;
        address_of_previous_link = &current.next;
    }
    
    // Most of the proc is COPIED☠️  from toDoublyLinkedList.
    // The only difference is this:
    current.next = result.head;
    result.head.previous = current;

    return;
}

CDLL_print :: proc(l: CircularDoublyLinkedList($T)) {
    n := l.head;
    for i := 0; i < l.length; i += 1 {
        if n.next == nil do print(n.value);
        else do printf("%v, ", n.value);
        n = n.next;
    }
}

// Entirely COPIED☠️  from DLL_insert
CDLL_insert :: proc(l: ^CircularDoublyLinkedList($T), index: int, v: T) {
    assert(index >= 0 && index <= l.length, "CircularDoublyLinkedList insert out of bounds.");

    link_to_current := &l.head;
    current := l.head;
    previous: ^DoublyLinkedListNode(T) = nil;

    for i := 0; i != index; i += 1 {
        link_to_current = &current.next;
        previous = current;
        current = current.next;
    }

    link_to_current^ = new_clone(DoublyLinkedListNode(T){v, previous, current});
    if link_to_current^.next != nil do link_to_current^.next.previous = link_to_current^;

    l.length += 1;
}

CDLL_insertAfter :: proc(
    l: ^CircularDoublyLinkedList($T), node: ^DoublyLinkedListNode(T), value: T
) -> ^DoublyLinkedListNode(T) {
    new_node := new_clone(DoublyLinkedListNode(T){value, node, node.next});
    node.next.previous = new_node;
    node.next = new_node;

    l.length += 1;
    return new_node;
}

CDLL_append :: proc(l: ^CircularDoublyLinkedList($T), v: T) {
    if l.head == nil {
        l.head = new(DoublyLinkedListNode(T));
        l.head.previous, l.head.next = l.head, l.head;
        l.head.value = v;
    } else {
        new_node := new_clone(DoublyLinkedListNode(T){v, l.head.previous, l.head});
        l.head.previous.next = new_node;
        l.head.previous = new_node;
    }

    l.length += 1;
}

CDLL_clear :: proc(l: ^CircularDoublyLinkedList($T)) {
    current := l.head;
    for i in 0..l.length-1 {
        next := current.next;
        free(current);
        current = next;
    }

    l.head = nil;
    l.length = 0;
}

CDLL_toSlice :: proc(l: CircularDoublyLinkedList($T)) -> []T {
    result := make([]T, l.length);

    current := l.head;
    for i in 0..l.length-1 {
        result[i] = current.value;
        current = current.next;
    }

    return result;
}
