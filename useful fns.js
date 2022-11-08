function sum (list) {
        return is_null(list)
                ? 0
                : is_number(head(list))
                ? head(list) + sum(tail(list))
                : sum(head(list))+ sum(head(tail(list)));
    }

sum(list(1,2,3,4,5));

function fib(n) { 
    return n <= 1
            ? n
            : fib(n - 1) + fib(n - 2);
}

// flatten list 
function flatten_list(xs){
    return is_null(xs)
        ? xs
        : is_list(head(xs))
        ? append(flatten_list(head(xs)), flatten_list(tail(xs)))
        : append(list(head(xs)), flatten_list(tail(xs)));
}

// nCr
function choose(n, k) {
    return k > n
           ? 0
           : k === 0 || k === n
           ? 1
           : choose(n - 1, k) + choose(n - 1, k - 1);
}

choose(12, 6);


// nCr using mem
const mem = [];

function read(n, k) {
    return mem[n] === undefined
           ? undefined
           : mem[n][k];
}

function write(n, k, value) {
    if (mem[n] === undefined) {
        mem[n] = [];
    }
    mem[n][k] = value;
}

function mchoose(n, k) {
    if (read(n, k) !== undefined) {
        return read(n, k);
    } else {
        const result = k > n
                       ? 0
                       : k === 0 || k === n
                       ? 1
                       : mchoose(n - 1, k) + mchoose(n - 1, k - 1);
        write(n, k, result);
        return result;
    }
}

// mchoose(6, 1);
//mchoose(100, 50);

// Tests whether arrays A and B are structurally equal.
function equal_array(A, B) {
    if (!is_array(A) || !is_array(B)) {
        return false;
    } else if (array_length(A) !== array_length(B)) {
        return false;
    } else {
        let is_equal = true;
        const len = array_length(A);
        for (let i = 0; is_equal && i < len; i = i + 1) {
            if (is_array(A[i]) || is_array(B[i])) {
                is_equal = equal_array(A[i], B[i]);
            } else {
                is_equal = equal(A[i], B[i]);
            }
        }
        return is_equal;
    }
}
// swap elements in a matrix (destructive functions, modifies matrix)
function swap(A, i, j) {
    const temp = A[i];
    A[i] = A[j];
    A[j] = temp;
}
// const A = [1,2,3];
// swap(A,1,0);
// A;
//---------------------------------------------------------------
function copy_array(A) {
    const len = array_length(A);
    const B = [];
    for (let i = 0; i < len; i = i + 1) {
        B[i] = A[i];
    }
    return B;
}
//---------------------------------------------------------------
function enum_array(a, b) {
    const arr = [];
    const len = b - a + 1;
    let num = a;
    for (let i = 0; i < len; i = i + 1) {
        arr[i] = num;
        num = num + 1;
    }
    return arr;
    // returns [] if a > b
}
//---------------------------------------------------------------
// Non in-place transpose for m x n
function transpose_matrix(M) {
    const rows = array_length(M);
    const cols = array_length(M[0]);
    let new_M = [];
    
    for (let new_r = 0; new_r < cols; new_r = new_r + 1){
        new_M[new_r] = [];
        for (let new_c = 0; new_c < rows; new_c = new_c + 1){
            new_M[new_r][new_c] = M[new_c][new_r];
        }
    }
    return new_M;
}

const mat = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]];
rotate_matrix(mat);
mat;
//---------------------------------------------------------------
// In-place transpose for n x n 
function transpose_matrix(M) {
    const n = array_length(M); // M is assumed n x n
    function swap(r1, c1, r2, c2) {
        const temp = M[r1][c1];
        M[r1][c1] = M[r2][c2];
        M[r2][c2] = temp;
    }
    for (let r = 0; r < n; r = r + 1) {
        for (let c = r + 1; c < n; c = c + 1) {
            swap(r, c, c, r);
        }
    }
}
//---------------------------------------------------------------
function rotate_matrix(M) {
    const n = array_length(M); // M is assumed n x n
    function swap(r1, c1, r2, c2) {
        const temp = M[r1][c1];
        M[r1][c1] = M[r2][c2];
        M[r2][c2] = temp;
    }
    // Do a matrix transpose first.
    transpose_matrix(M);
    
    // Then reverse each row. (For clockwise rotate)
    const half_n = math_floor(n / 2);
    for (let r = 0; r < n; r = r + 1) {
        for (let c = 0; c < half_n; c = c + 1) {
            swap(r, c, r, n - c - 1);
        }
    }
    // For anticlockwise rotation (DELETE IF UNNECESSARY)
    const half_n = math_floor(n / 2);
    for (let r = 0; r < half_n; r = r + 1) {
        for (let c = 0; c < n; c = c + 1) {
            swap(r, c, n - r - 1, c);
        }
    }
}
//---------------------------------------------------------------
function matrix_multiply_nxn(n, A, B) {
    const M = [];
    for (let r = 0; r < n; r = r + 1) {
        M[r] = [];
        for (let c = 0; c < n; c = c + 1) {
            M[r][c] = 0;
            for (let k = 0; k < n; k = k + 1) {
                M[r][c] = M[r][c] + A[r][k] * B[k][c];
            }
        }
    }
    return M;
}
//---------------------------------------------------------------
function reverse_array(A) {
    const len = array_length(A);
    const half_len = math_floor(len / 2);
    for (let i = 0; i < half_len; i = i + 1) {
        swap(A, i, len - 1 - i);
    }
}
//---------------------------------------------------------------
function array_to_list(A) {
    const len = array_length(A);
    let L = null;
    for (let i = len - 1; i >= 0; i = i - 1) {
        L = pair(A[i], L);
    }
    return L;
}
//---------------------------------------------------------------
function list_to_array(L) {
    const A = [];
    let i = 0;
    for (let p = L; !is_null(p); p = tail(p)) {
        A[i] = head(p);
        i = i + 1;
    }
    return A;
}
//---------------------------------------------------------------
// Sorts the array of numbers in ascending order.
function sort_ascending(A) {
    const len = array_length(A);
    for (let i = 1; i < len; i = i + 1) {
        const x = A[i];
        let j = i - 1;
        while (j >= 0 && A[j] > x) {
            A[j + 1] = A[j];
            j = j - 1;
        }
        A[j + 1] = x;
    }
}
//---------------------------------------------------------------
function digits_to_string(digits) {
    const len = array_length(digits);
    let str = "";
    for (let i = 0; i < len; i = i + 1) {
        str = str + stringify(digits[i]);
    }
    return str;
}
// const D = [8, 3, 9, 2, 8, 1];
// digits_to_string(D);  // returns "839281"


function permutations(ys) {
        return is_null(ys)
            ? list(null)
            : accumulate(append, null,
                map(x => map(p => pair(x, p),
                             permutations(remove(x, ys))),
                    ys));
    }
function a_subsets(xs) {
    return accumulate(
                (x, ss) => append(ss, map(s => pair(x, s), ss)),
                list(null),
                xs);
} //using accumulate
function subsets(xs) {
    if (is_null(xs)) {
        return list(null);
    } else {
        const subsets_rest = subsets(tail(xs));
        const x = head(xs);
        const has_x = map(s => pair(x, s), subsets_rest);
        return append(subsets_rest, has_x);
    }
}
function a_remove_duplicates(lst) {
    return accumulate(
            (x, xs) =>
                is_null(member(x, xs))
                    ? pair(x, xs)
                    : xs,
            null,
            lst);
} //using accumulate
function remove_duplicates(lst) {
    // returns a set (list of distinct objects)
    return is_null(lst)
        ? null
        : pair(
            head(lst),
            remove_duplicates(
                filter(x => !equal(x, head(lst)), tail(lst))));
}

function all_different(nums) {
    if (is_null(nums)) {
        return true;
    } else {
        let head_is_unique = is_null(member(head(nums), tail(nums)));
        return head_is_unique && all_different(tail(nums));
    }
} // boolean i think
function count_pairs(x) {
    let pairs = null;
    function check(y) {
        if (!is_pair(y)) {
            return undefined;
        } else if (!is_null(member(y, pairs))) {
            return undefined;
        } else {
            pairs = pair(y, pairs);
            check(head(y));
            check(tail(y));
        }
    }
    check(x);
    return length(pairs);
}
function skip (L, n) {
            return n===len //len >= 2
                    ? tail(tail(L))
                    : skip(tail(L), n+1);
        } // skips first n elements of L
function repeat (L) {
                   return is_null(tail(L))
                        ? null 
                        : head(L) === head(tail(L))
                        ? pair(head(L), repeat(tail(L)))
                        : null;  
            } // returns a list of numbers that is repeated consecutively in L

// insertion sort: 
// Sort the tail of the given list using wishful thinking! 
// Insert the head in the right place
function insert(x, xs) { 
    return is_null(xs)
        ? list(x)
        : x <= head(xs)
        ? pair(x, xs)
        : pair(head(xs), insert(x, tail(xs)));
}
function insertion_sort(xs) { 
    return is_null(xs)
            ? xs
            : insert(head(xs), insertion_sort(tail(xs)));
}

// selection sort:
// Find the smallest element x and remove it from the list 
// Sort the remaining list, and put x in front

  // find smallest element of non-empty list xs
function smallest(xs) {
    return accumulate((x, y) => x < y ? x : y,
                         head(xs), tail(xs));
}
function selection_sort(xs) {
    if (is_null(xs)) { 
        return xs;
    } else {
    const x = smallest(xs);
    return pair(x, selection_sort(remove(x, xs)));
    } 
    
}

// merge sort:
// Split the list in half, sort each half using wishful thinking 
// Merge the sorted lists together
function merge_sort(xs) {
    if (is_null(xs) || is_null(tail(xs))) {
        return xs; 
        
    } else {
        const mid = middle(length(xs));
        return merge(merge_sort(take(xs, mid)),
                        merge_sort(drop(xs, mid)));
    } 
    
}

function merge(xs, ys) { 
    if (is_null(xs)) {
        return ys;
    } else if (is_null(ys)) {
        return xs;  
    } else {
        const x = head(xs); const y = head(ys); 
        return x < y
                ? pair(x, merge(tail(xs), ys))  
                : pair(y, merge(xs, tail(ys)));
    }
}
// half, rounded downwards
function middle(n) {
return math_floor(n / 2);
}
   // put the first n elements of xs into a list
function take(xs, n) { 
    function take_helper(xs, start, stop) {
        return start === stop
            ? null
            : pair(list_ref(xs, start), take_helper(xs, start + 1, stop));
  }

    return take_helper(xs, 0, n);
}
// drop the first n elements from the list // and return the rest
function drop(xs, n) {
    return n === 0 ? xs : drop(tail(xs), n - 1);
}

// destructive append modifies original lists
function d_append(xs, ys) { 
    if (is_null(xs)) {
        return ys; 
    } else {
        set_tail(xs, d_append(tail(xs), ys));
        return xs; 
    }
}

// destructive map
function d_map(fun, xs) { 
    if (!is_null(xs)) {
        set_head(xs, fun(head(xs)));
        d_map(fun, tail(xs)); 
    } else { }
}

//linear search (boolean)
function linear_search(A, v) {
    const len = array_length(A);
    let i = 0;
    while (i < len && A[i] !== v) {
        i = i + 1;
    }
    return (i < len);
}

//linear_search([1,2,3,4,5,6,7,8,9], 5);

// binary search using recursion
function binary_Rsearch(A, v) {
function search(low, high) { 
    if (low > high) {
return false; } else {
const mid = math_floor((low + high) / 2); return (v === A[mid]) ||
(v < A[mid]
? search(low, mid - 1)
: search(mid + 1, high));
return search(0, array_length(A) - 1); }

//binary_Rsearch([1,2,3,4,5,6,7,8,9], 8);

// binary search using loops
function binary_Lsearch(A, v) {
    let low = 0;
    let high = array_length(A) - 1;

    while (low <= high) {
        const mid = math_floor((low + high) / 2 );
        if (v === A[mid]) {
            break;
        } else if (v < A[mid]) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return (low <= high);
}

//binary_Lsearch([1,2,3,4,5,6,7,8,9], 8);

//selection sort for arrays 
function selection_sortA(A) {
    const len = array_length(A);

    for (let i = 0; i < len - 1; i = i + 1) {
        let min_pos = find_min_pos(A, i, len - 1);
        swap(A, i, min_pos);
    }
}

function find_min_pos(A, low, high) {
    let min_pos = low;
    for (let j = low + 1; j <= high; j = j + 1) {
        if (A[j] < A[min_pos]) {
            min_pos = j;
        }
    }
    return min_pos;
}

function swap(A, x, y) {
    const temp = A[x];
    A[x] = A[y];
    A[y] = temp;
}

// const A = [3, 9, 2, 1, 6, 5, 3, 8];
// selection_sort(A);
// A;

//insertion sort for arrays 
function insertion_sortA(A) {
    const len = array_length(A);
    
    for (let i = 1; i < len; i = i + 1) {
        let j = i - 1;
        while (j >= 0 && A[j] > A[j + 1]) {
            swap(A, j, j + 1);
            j = j - 1;
        }
    }
}

function swap(A, x, y) {
    const temp = A[x];
    A[x] = A[y];
    A[y] = temp;
}
// const A = [3, 9, 2, 1, 6, 5, 3, 8];
// insertion_sort(A);
// A;

// merge sort for arrays
function merge_sortA(A) {
    merge_sort_helper(A, 0, array_length(A) - 1);
}

function merge_sort_helper(A, low, high) {
    if (low < high) {
        const mid = math_floor((low + high) / 2);
        merge_sort_helper(A, low, mid);
        merge_sort_helper(A, mid + 1, high);
        merge(A, low, mid, high);
    }
}

function mergeA(A, low, mid, high) {
    const B = [];
    let left = low;
    let right = mid + 1;
    let Bidx = 0;
    
    while (left <= mid && right <= high) {
        if (A[left] <= A[right]) {
            B[Bidx] = A[left];
            left = left + 1;
        } else {
            B[Bidx] = A[right];
            right = right + 1;
        }
        Bidx = Bidx + 1;
    }
    
    while (left <= mid) {
        B[Bidx] = A[left];
        Bidx = Bidx + 1;
        left = left + 1;
    }   
    while (right <= high) {
        B[Bidx] = A[right];
        Bidx = Bidx + 1;
        right = right + 1;
    }
    
    for (let k = 0; k < high - low + 1; k = k + 1) {
        A[low + k] = B[k];
    }
}

// const A = [3, 9, 2, 1, 6, 5, 3, 8];
// merge_sortA(A);
// A;

//BAE
// Function to calculate arithmetic ops from list(op, n1, n2) form 
function normalise(xs){
    
    function apply(f, n1, n2){
        return f === '+'
            ? n1 + n2
            : f === '-'
            ? n1 - n2
            : f === '*'
            ? n1 * n2
            : n1 / n2;
    }
    
    if (is_number(xs)){
	return xs;
    }

    const op = head(xs);
    let left = normalise(head(tail(xs)));
    let right = normalise(head(tail(tail(xs))));

    return apply(op, left, right);
}
//normalise(list('+', 1, list('*', 5, 7)));

function evaluate_BAE_tree(bae_tree) {
    // WRITE HERE.
    if (is_list(bae_tree)) {
        const left = evaluate_BAE_tree(head(bae_tree));
        const right = evaluate_BAE_tree(head(tail(tail(bae_tree))));
        const op = head(tail(bae_tree));
        if (op === "+") {
            return left + right;
        } else if (op === "-") {
            return left - right;
        } else if (op === "*") {
            return left * right;
        } else { // (op === "/")
            return left / right;
        }
    } else { // is a number
        return bae_tree;
    }
}
const bae_tree = list(list(1, "/", 2), "+", 5);
evaluate_BAE_tree(bae_tree); // returns 123

function build_BAE_tree (bae_list) {
    return  is_null(bae_list)
            ? null
            : is_number(head(bae_list))
            ? head(bae_list)
            : head(bae_list) === "(" || head(bae_list) === ")"
            ? pair(pair((build_BAE_tree(tail(bae_list))), null), null)
            : pair(head(bae_list), build_BAE_tree(tail(bae_list)));
}
//-------------------------------------------------------------------------------
function equalnumber (paren_list) {
        if(is_null(paren_list)) {
            return true;
        } else if (length(paren_list) %2!==0) {
            return false;
        } else {
            if (!is_null(member("(", paren_list)) && !is_null(member(")", paren_list))) {
                const x = remove("(", paren_list);
                const y = remove(")", x);
                return equalnumber(y);
            } else {
                return false;
            }
        }
    } // check equal number of paranthesis
    
const paren_list = list("(", "(", ")", ")"); 
equalnumber(paren_list);
// returns true

function order(paren_list) {
    let len = length(paren_list) -1;
    return list_ref(paren_list, len) === "(" || list_ref(paren_list, 0) === ")"
            ? false
            : true;
} // check order of parathensis

const t = list("(","(",")","(");
order(t);
    
function check_parentheses (paren_list) {
    return is_null(paren_list)
            ? true
            : equalnumber(paren_list) && order(paren_list);
}

// tree 
//scale tree
function scale_tree(tree , factor) {
    return map(
            sub_tree =>
                ! is_list(sub_tree)
                    ? factor * sub_tree
                    : scale_tree(sub_tree , factor),
            tree);
}
// map tree 
function map_tree(f, tree) {
    return map(
            sub_tree =>
                ! is_list(sub_tree)
                    ? f(sub_tree)
                    : map_tree(f, sub_tree),
            tree);
}

// accumulate on tree
function accumulate_tree(f1, f2, initial, tree) {
    return is_null(tree)
        ? initial
        : f2( is_list(head(tree))
            ? accumulate_tree(f1, f2, initial, head(tree))
            : f1(head(tree)),
                accumulate_tree(f1, f2, initial, tail(tree)));
}

// length of tree 
function count_data_items(tree) {
    return is_null(tree)
        ? 0
        : ( is_list(head(tree))
            ? count_data_items(head(tree))
            : 1 )
            +
            count_data_items(tail(tree));
}

// deep flattening; follows similar structure 
function flatten(tree) {
return accumulate((x,y) => is_list(x) ? append(flatten(x), y)
: pair(x,y), null, tree);
// filter on tree
function filter_tree(pred, tree) {
return accumulate((x,y) => is_tree(x)
? pair(filter_tree(pred, x), y) : pair(x, y), null, tree);


//Counts the number of matches between 2 list of numbers
function num_of_matches(numsA, numsB) { function helper(xs, ys) {
return map(x => !is_null(member(x, ys)), xs); }
const r = helper(numsA, numsB);
return accumulate((x,y) => x ? y + 1 : y, 0, r);

/* Streams stuff */
// convert binary operation into stream binary operation function extend(bin) {
const res = (str1, str2) =>
pair(bin(head(str1), head(str2)),
() => res(stream_tail(str1), stream_tail(str2)));
return res;
// stream_append_pickle but interleaving form
function stream_interleaving_append_pickle(xs, ys) {
    return is_null(xs)
    ? ys()
    : pair(head(xs),
            () => stream_append_pickle(ys(), 
                () => stream_tail(xs)));
}
// infinite stream generation (normal and wishful thinking)
/* TEMPLATE RELATED MATTERS #4 (normal infinite streams) */
const ones = pair(1, () => ones);
const alt_ones = pair(-1, () => pair(1, () => alt_ones));
/* TEMPLATE RELATED MATTERS #5 (normal infinite streams with arg passing) */

// replace every occurrence of element a in stream with b 
function replace(s, a, b) {
    return is_null(s)
        ? null
        : pair((head(s) === a) ? b : head(s), () => replace(stream_tail(s), a, b));
// returns inreasing sequence with step 1 as b keeps increasing
function more(a, b) { 
    return (a > b)
            ? more(1, 1 + b)
            : pair(a, () => more(a + 1, b));
}
/* TEMPLATE RELATED MATTERS #6 (wishful thinking on streams) */ 
function sum_stream(stream) {
    return pair(head(stream), 
                () => add_streams(sum_stream(stream), stream_tail(stream)));
}

// fibs using streams
const fibs = pair(0, () => pair(1, () => add_streams(stream_tail(fibs), fibs))); 
// generate a sequence with the same equation as fibonacci but with different initial values
function fibgen(a, b) {
    return pair(a, () => fibgen(b, a + b));
}

function sieve(s) {
    return pair(head(s),
            () => sieve(stream_filter(x => !is_divisible(x, head(s)), stream_tail(s))));
}

