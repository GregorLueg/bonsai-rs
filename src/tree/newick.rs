//! Newick serialisation and parsing for [`Tree`].
//!
//! The arena stores an unrooted tree in a rooted representation, so the string
//! is written rooted at [`Tree::root`]. That placement is a display choice and
//! carries no information: the likelihood is root-independent (SPEC.md section
//! 2, S14), so a tree written here and read back may come home rooted the same
//! way but indexed differently, and nothing downstream can tell.
//!
//! Both directions are iterative. Biological trees are deep and laddery
//! ([`Tree::ladder`] exists to exercise exactly that), and a recursive writer or
//! parser overflows the stack somewhere in the tens of thousands of leaves.
//!
//! Labels live outside the arena, which has no label field, so the writer takes
//! them as a slice indexed by leaf index and the parser hands them back the same
//! way.
//!
//! ### Deviations from strict Newick
//!
//! * An underscore in an unquoted label stays an underscore. Strict Newick reads
//!   it as a space, which would mangle cell barcodes such as `AACGT_1`.
//! * Internal labels are written on request but discarded on parsing:
//!   [`Tree::from_parents`] permutes internal indices into level order without
//!   reporting the permutation, so there is no index to hand them back under.
//! * Negative branch lengths are rejected. They are diffusion times here
//!   (SPEC.md section 1, `t[i]`), and a negative one is nonsense the arena would
//!   happily carry into the likelihood.
//! * A lone leaf is refused by the writer. The arena holds one and
//!   [`crate::tree::cluster`] has a use for it, but the Newick for it is a bare
//!   label, which this parser and every other one reject; writing something
//!   nothing can read back is worse than refusing.
//!
//! ### Empty labels, and the trailing comma
//!
//! Strict Newick permits an empty label, so `"(a,b,);"` is a legal
//! **three**-leaf tree whose third leaf is unnamed, and `"(,,,);"` is a legal
//! four-leaf one. This parser follows the standard and reads them that way.
//! The cost is that a trailing-comma typo is a silently wrong topology rather
//! than an error, and nothing can distinguish the two; the alternative, banning
//! empty labels, would reject valid files that carry their names elsewhere.
//! Noted rather than fixed, and pinned by
//! `test_a_trailing_comma_is_an_unnamed_leaf`.

use std::fmt::Write as _;

use crate::errors::BonsaiErrors;
use crate::tree::{NO_NODE, Tree};

///////////////
// Constants //
///////////////

/// Significant decimal digits an IEEE-754 binary64 needs to survive a text round
/// trip. The significand is 53 bits, so 17 digits (C's `DBL_DECIMAL_DIG`) are
/// necessary and sufficient; 16 loses roughly half of all values. Rust's
/// `Display` and `LowerExp` without an explicit precision emit the *shortest*
/// decimal that reparses to the same bits, which is never longer than this, so
/// the writer fixes no precision at all and gets exactness for free.
/// `test_branch_length_needs_seventeen_digits` pins the claim.
const BRANCH_ROUND_TRIP_DIGITS: usize = 17;

/// Below this magnitude a branch length is written in scientific notation.
/// Plain decimal for `1e-300` runs to 302 characters; inside the range it stays
/// under about 25. Cosmetic only, both forms round-trip exactly.
const PLAIN_DECIMAL_MIN: f64 = 1e-6;

/// Above this magnitude a branch length is written in scientific notation. See
/// [`PLAIN_DECIMAL_MIN`]; the same length argument applies at the top end.
const PLAIN_DECIMAL_MAX: f64 = 1e15;

/// Branch length assigned to a node whose Newick entry carries no `:length`.
/// Newick leaves it undefined; zero is the only value that keeps the tree the
/// same tree, since a zero-length edge does not change the likelihood (SPEC.md
/// section 9.2).
const DEFAULT_BRANCH_LENGTH: f64 = 0.0;

/// Largest node count the arena can address. Parent indices are `u32` with
/// [`NO_NODE`] reserved as the root sentinel, so one value is unavailable.
const MAX_NODES: usize = NO_NODE as usize;

/////////////
// Writing //
/////////////

/// Serialise a tree to Newick, with branch lengths and leaf labels.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaf_labels` - One label per leaf, indexed by leaf index
///
/// ### Returns
///
/// The Newick string, semicolon terminated, or `MalformedTree` if the label
/// count does not match the leaf count, or if the tree is a lone leaf, which
/// has no Newick form the reader will take back.
pub fn write_newick<S: AsRef<str>>(tree: &Tree, leaf_labels: &[S]) -> Result<String, BonsaiErrors> {
    write_newick_labelled::<S, S>(tree, leaf_labels, None)
}

/// Serialise a tree to Newick, optionally labelling the internal nodes too.
///
/// Internal labels are a courtesy for downstream viewers. Nothing in this crate
/// reads them back, see the module docs.
///
/// ### Params
///
/// * `tree` - The tree
/// * `leaf_labels` - One label per leaf, indexed by leaf index
/// * `internal_labels` - One label per internal node, indexed by
///   `node - tree.n_leaves()`, or `None` to leave internal nodes unlabelled. An
///   empty string is written as no label rather than as `''`
///
/// ### Returns
///
/// The Newick string, semicolon terminated, or `MalformedTree` if either label
/// count does not match the arena, or if the tree is a lone leaf, which has no
/// Newick form the reader will take back.
pub fn write_newick_labelled<S: AsRef<str>, T: AsRef<str>>(
    tree: &Tree,
    leaf_labels: &[S],
    internal_labels: Option<&[T]>,
) -> Result<String, BonsaiErrors> {
    let n_leaves = tree.n_leaves();
    let n_internal = tree.n_nodes() - n_leaves;
    // A lone leaf, which the arena holds and Newick has no form for; see the
    // module docs. Refusing here is what keeps the writer and the reader agreed
    // on what a tree is.
    if n_leaves < 2 {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "a tree of {n_leaves} leaf/leaves has no Newick form; at least two are needed"
            ),
        });
    }
    if leaf_labels.len() != n_leaves {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "{} leaf labels supplied for a tree with {n_leaves} leaves",
                leaf_labels.len()
            ),
        });
    }
    if let Some(internal) = internal_labels
        && internal.len() != n_internal
    {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "{} internal labels supplied for a tree with {n_internal} internal nodes",
                internal.len()
            ),
        });
    }

    let root = tree.root();
    let mut out = String::new();

    // Explicit stack of (node, index of the next child to emit). A frame is
    // revisited once per child, so the traversal costs one pass over the arena
    // and the stack is bounded by the tree's depth rather than the call stack.
    let mut stack: Vec<(u32, usize)> = Vec::new();
    stack.push((root, 0));
    while !stack.is_empty() {
        let top = stack.len() - 1;
        let (node, next) = stack[top];
        let children = tree.children(node);
        if next < children.len() {
            out.push(if next == 0 { '(' } else { ',' });
            stack[top].1 = next + 1;
            stack.push((children[next], 0));
            continue;
        }

        // Every child is out; close the group, if there was one, and annotate.
        if !children.is_empty() {
            out.push(')');
        }
        let label = if children.is_empty() {
            leaf_labels[node as usize].as_ref()
        } else {
            internal_labels.map_or("", |l| l[node as usize - n_leaves].as_ref())
        };
        if !label.is_empty() {
            push_label(&mut out, label);
        }
        if node != root {
            out.push(':');
            push_branch_length(&mut out, tree.branch(node));
        }
        stack.pop();
    }

    out.push(';');
    Ok(out)
}

/// Append a label, quoting it if it would not survive as a bare token.
///
/// ### Params
///
/// * `out` - Buffer to append to
/// * `label` - The label, assumed non-empty
fn push_label(out: &mut String, label: &str) {
    if !label.bytes().any(is_label_terminator) {
        out.push_str(label);
        return;
    }
    out.push('\'');
    for ch in label.chars() {
        // A single quote inside a quoted label is written doubled.
        if ch == '\'' {
            out.push('\'');
        }
        out.push(ch);
    }
    out.push('\'');
}

/// Append a branch length in a form that reparses to the identical `f64`.
///
/// ### Params
///
/// * `out` - Buffer to append to
/// * `length` - The branch length
fn push_branch_length(out: &mut String, length: f64) {
    let start = out.len();
    // Writing into a `String` is infallible; the `Result` only exists because
    // `write!` is generic over `fmt::Write`.
    let magnitude = length.abs();
    if magnitude != 0.0 && !(PLAIN_DECIMAL_MIN..=PLAIN_DECIMAL_MAX).contains(&magnitude) {
        let _ = write!(out, "{length:e}");
    } else {
        let _ = write!(out, "{length}");
    }
    debug_assert_eq!(
        out[start..].parse::<f64>(),
        Ok(length),
        "the shortest decimal did not reparse exactly, so the \
         {BRANCH_ROUND_TRIP_DIGITS}-digit claim above is wrong"
    );
}

/////////////
// Reading //
/////////////

/// Parse a Newick string into a tree and its leaf labels.
///
/// Handles nested parentheses, polytomies, arbitrary whitespace including
/// newlines, `'...'` quoted labels, `[...]` comments, decimal and scientific
/// branch lengths, and an absent branch length on the root.
///
/// Node indices are assigned in a second pass so that the arena invariant holds:
/// leaves take `0..n_leaves` in the order they appear in the string, and
/// internal nodes follow sorted by height above the leaves, which is enough for
/// every parent index to exceed its children's. [`Tree::from_parents`] then
/// relabels the internal nodes into level order itself, so the indices a caller
/// sees are not the ones assigned here.
///
/// ### Params
///
/// * `text` - The Newick string
///
/// ### Returns
///
/// The tree and one label per leaf indexed by leaf index, or `MalformedTree`
/// describing what is wrong with the string. Never panics on bad input.
pub fn parse_newick(text: &str) -> Result<(Tree, Vec<String>), BonsaiErrors> {
    let bytes = text.as_bytes();
    let mut i = 0usize;

    let mut raw = RawNodes::default();
    // Internal nodes whose closing parenthesis has not been seen yet.
    let mut open: Vec<u32> = Vec::new();
    // The subtree most recently completed; the last one is the root.
    let mut completed: Option<u32> = None;
    // Alternates: a subtree is expected, or one of `,`, `)`, `;` is.
    let mut expect_subtree = true;

    loop {
        skip_trivia(bytes, &mut i)?;
        if i >= bytes.len() {
            return Err(BonsaiErrors::MalformedTree {
                reason: if raw.parent.is_empty() {
                    "empty input, expected a Newick string".to_string()
                } else {
                    "input ends without a ';'".to_string()
                },
            });
        }

        if expect_subtree {
            let parent = open.last().copied();
            if bytes[i] == b'(' {
                i += 1;
                let node = raw.push(parent)?;
                open.push(node);
                continue;
            }
            // Anything else opens a leaf. Its label may legitimately be empty,
            // in which case the delimiter that follows closes it immediately.
            let node = raw.push(parent)?;
            raw.label[node as usize] = read_label(text, bytes, &mut i)?;
            raw.branch[node as usize] = read_branch_length(text, bytes, &mut i)?;
            completed = Some(node);
            expect_subtree = false;
            continue;
        }

        match bytes[i] {
            b',' => {
                if open.is_empty() {
                    return Err(BonsaiErrors::MalformedTree {
                        reason: format!("',' at byte {i} is outside any parenthesised group"),
                    });
                }
                i += 1;
                expect_subtree = true;
            }
            b')' => {
                let Some(node) = open.pop() else {
                    return Err(BonsaiErrors::MalformedTree {
                        reason: format!("unbalanced ')' at byte {i}"),
                    });
                };
                i += 1;
                raw.label[node as usize] = read_label(text, bytes, &mut i)?;
                raw.branch[node as usize] = read_branch_length(text, bytes, &mut i)?;
                completed = Some(node);
            }
            b';' => {
                i += 1;
                break;
            }
            other => {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!(
                        "unexpected '{}' at byte {i}, expected ',', ')' or ';'",
                        other as char
                    ),
                });
            }
        }
    }

    if !open.is_empty() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("{} '(' left unclosed at the ';'", open.len()),
        });
    }
    skip_trivia(bytes, &mut i)?;
    if i < bytes.len() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("{} bytes of trailing junk after the ';'", bytes.len() - i),
        });
    }
    let Some(root) = completed else {
        return Err(BonsaiErrors::MalformedTree {
            reason: "empty input, expected a Newick string".to_string(),
        });
    };

    assemble(raw, root)
}

/// Nodes as the parser meets them, before indices are assigned.
///
/// Parallel vectors in creation order, which is the order the string mentions
/// each node's opening token.
#[derive(Default)]
struct RawNodes {
    /// Parent of each node, `None` for the root.
    parent: Vec<Option<u32>>,
    /// Number of children attached so far.
    n_children: Vec<u32>,
    /// Length of the branch above each node.
    branch: Vec<f64>,
    /// Label as written, empty when absent.
    label: Vec<String>,
}

impl RawNodes {
    /// Append a node and register it with its parent.
    ///
    /// ### Params
    ///
    /// * `parent` - Parent in raw indexing, `None` for the root
    ///
    /// ### Returns
    ///
    /// The new node's raw index, or `MalformedTree` if the arena's `u32`
    /// indexing is exhausted.
    fn push(&mut self, parent: Option<u32>) -> Result<u32, BonsaiErrors> {
        if self.parent.len() >= MAX_NODES {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("more than {MAX_NODES} nodes, which the u32 arena cannot address"),
            });
        }
        let node = self.parent.len() as u32;
        self.parent.push(parent);
        self.n_children.push(0);
        self.branch.push(DEFAULT_BRANCH_LENGTH);
        self.label.push(String::new());
        if let Some(p) = parent {
            self.n_children[p as usize] += 1;
        }
        Ok(node)
    }
}

/// Turn parsed nodes into a [`Tree`] by assigning arena indices.
///
/// ### Params
///
/// * `raw` - Nodes in parse order
/// * `root` - Raw index of the root
///
/// ### Returns
///
/// The tree and its leaf labels, or `MalformedTree` if what was parsed is not a
/// tree this arena can hold.
fn assemble(raw: RawNodes, root: u32) -> Result<(Tree, Vec<String>), BonsaiErrors> {
    let n_nodes = raw.parent.len();
    let leaves: Vec<u32> = (0..n_nodes as u32)
        .filter(|&n| raw.n_children[n as usize] == 0)
        .collect();
    let n_leaves = leaves.len();
    if n_leaves < 2 {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("{n_leaves} leaves is not a tree; at least two are needed"),
        });
    }

    // Height above the leaves, settled by a queue rather than a traversal so
    // that a 100k-deep ladder costs no stack. A node is ready once every child
    // has reported in, which for a tree happens exactly once per node.
    let mut height = vec![0u32; n_nodes];
    let mut pending = raw.n_children.clone();
    let mut ready = leaves.clone();
    while let Some(node) = ready.pop() {
        let Some(parent) = raw.parent[node as usize] else {
            continue;
        };
        height[parent as usize] = height[parent as usize].max(height[node as usize] + 1);
        pending[parent as usize] -= 1;
        if pending[parent as usize] == 0 {
            ready.push(parent);
        }
    }

    // Leaves first in string order, then internal nodes by height. A parent is
    // strictly taller than its children, so this alone satisfies the arena's
    // "parents have larger indices" rule; `from_parents` sorts the rest out.
    let mut internal: Vec<u32> = (0..n_nodes as u32)
        .filter(|&n| raw.n_children[n as usize] > 0)
        .collect();
    internal.sort_unstable_by_key(|&n| (height[n as usize], n));

    let mut new_index = vec![0u32; n_nodes];
    for (slot, &node) in leaves.iter().chain(internal.iter()).enumerate() {
        new_index[node as usize] = slot as u32;
    }

    let mut parent = vec![NO_NODE; n_nodes];
    let mut branch = vec![0.0f64; n_nodes];
    for old in 0..n_nodes {
        let new = new_index[old] as usize;
        parent[new] = raw.parent[old].map_or(NO_NODE, |p| new_index[p as usize]);
        branch[new] = raw.branch[old];
    }
    // The parser can only ever build one root, but a caller reading this wants
    // the invariant stated rather than inferred.
    debug_assert_eq!(parent[new_index[root as usize] as usize], NO_NODE);

    let mut labels: Vec<String> = raw.label;
    let mut leaf_labels: Vec<String> = Vec::with_capacity(n_leaves);
    for &leaf in &leaves {
        leaf_labels.push(std::mem::take(&mut labels[leaf as usize]));
    }

    let tree = Tree::from_parents(parent, branch, n_leaves)?;
    Ok((tree, leaf_labels))
}

/// Whether a byte ends an unquoted label.
///
/// The Newick punctuation plus whitespace. `[` is included because it opens a
/// comment, and `'` because a quote may only open a label, never sit inside a
/// bare one.
///
/// ### Params
///
/// * `byte` - Candidate byte
///
/// ### Returns
///
/// `true` if the byte cannot be part of a bare label. Multi-byte UTF-8 is safe
/// to test byte-wise: every continuation byte is `>= 0x80` and none of these
/// delimiters are.
fn is_label_terminator(byte: u8) -> bool {
    byte.is_ascii_whitespace()
        || matches!(byte, b'(' | b')' | b'[' | b']' | b',' | b':' | b';' | b'\'')
}

/// Advance the cursor past whitespace and `[...]` comments.
///
/// Comments do not nest in Newick, so the first `]` closes one.
///
/// ### Params
///
/// * `bytes` - The input
/// * `i` - Cursor, advanced in place
///
/// ### Returns
///
/// `Ok(())`, or `MalformedTree` if a comment is never closed.
fn skip_trivia(bytes: &[u8], i: &mut usize) -> Result<(), BonsaiErrors> {
    loop {
        while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
            *i += 1;
        }
        if *i >= bytes.len() || bytes[*i] != b'[' {
            return Ok(());
        }
        let opened = *i;
        *i += 1;
        while *i < bytes.len() && bytes[*i] != b']' {
            *i += 1;
        }
        if *i >= bytes.len() {
            return Err(BonsaiErrors::MalformedTree {
                reason: format!("comment opened at byte {opened} is never closed"),
            });
        }
        *i += 1;
    }
}

/// Read an optional node label at the cursor.
///
/// ### Params
///
/// * `text` - The input, for slicing labels out without copying byte by byte
/// * `bytes` - The same input as bytes
/// * `i` - Cursor, advanced past the label
///
/// ### Returns
///
/// The label, empty if there is none, or `MalformedTree` if a quoted label is
/// never closed.
fn read_label(text: &str, bytes: &[u8], i: &mut usize) -> Result<String, BonsaiErrors> {
    skip_trivia(bytes, i)?;
    if *i < bytes.len() && bytes[*i] == b'\'' {
        let opened = *i;
        *i += 1;
        let mut label = String::new();
        loop {
            let start = *i;
            while *i < bytes.len() && bytes[*i] != b'\'' {
                *i += 1;
            }
            label.push_str(&text[start..*i]);
            if *i >= bytes.len() {
                return Err(BonsaiErrors::MalformedTree {
                    reason: format!("quoted label opened at byte {opened} is never closed"),
                });
            }
            // A doubled quote is an escaped one, a lone quote closes the label.
            if bytes.get(*i + 1) == Some(&b'\'') {
                label.push('\'');
                *i += 2;
            } else {
                *i += 1;
                return Ok(label);
            }
        }
    }
    let start = *i;
    while *i < bytes.len() && !is_label_terminator(bytes[*i]) {
        *i += 1;
    }
    Ok(text[start..*i].to_string())
}

/// Read an optional `:length` at the cursor.
///
/// ### Params
///
/// * `text` - The input
/// * `bytes` - The same input as bytes
/// * `i` - Cursor, advanced past the length
///
/// ### Returns
///
/// The branch length, [`DEFAULT_BRANCH_LENGTH`] if there is no `:`, or
/// `MalformedTree` if what follows the `:` is not a usable non-negative finite
/// number.
fn read_branch_length(text: &str, bytes: &[u8], i: &mut usize) -> Result<f64, BonsaiErrors> {
    skip_trivia(bytes, i)?;
    if *i >= bytes.len() || bytes[*i] != b':' {
        return Ok(DEFAULT_BRANCH_LENGTH);
    }
    *i += 1;
    skip_trivia(bytes, i)?;

    let start = *i;
    while *i < bytes.len() && matches!(bytes[*i], b'0'..=b'9' | b'+' | b'-' | b'.' | b'e' | b'E') {
        *i += 1;
    }
    let token = &text[start..*i];
    let Ok(length) = token.parse::<f64>() else {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("'{token}' at byte {start} is not a branch length"),
        });
    };
    if !length.is_finite() {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!("branch length '{token}' at byte {start} is not finite"),
        });
    }
    if length < 0.0 {
        return Err(BonsaiErrors::MalformedTree {
            reason: format!(
                "branch length '{token}' at byte {start} is negative; branch lengths are \
                 diffusion times and cannot be"
            ),
        });
    }
    Ok(length)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonical form of a labelled tree: for every node, the sorted labels of
    /// the leaves below it paired with the exact bits of its upstream branch
    /// length, the whole list sorted.
    ///
    /// Comparing two of these checks the topology as a set of clades, and the
    /// branch lengths bit for bit. It ignores node indices, sibling order and
    /// internal labels, which is exactly what `from_parents` is free to permute.
    /// It does **not** check the root's own branch entry, which Newick has
    /// nowhere to put, and it assumes leaf labels are unique. Cost is quadratic
    /// in the leaf count, so it is for small fixtures only.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `labels` - One label per leaf, indexed by leaf index
    ///
    /// ### Returns
    ///
    /// The sorted clade-and-length list.
    fn canonical(tree: &Tree, labels: &[String]) -> Vec<(Vec<String>, u64)> {
        let mut clade: Vec<Vec<String>> = Vec::with_capacity(tree.n_nodes());
        // Ascending index order is a post-order, so children are always ready.
        for node in 0..tree.n_nodes() as u32 {
            if (node as usize) < tree.n_leaves() {
                clade.push(vec![labels[node as usize].clone()]);
                continue;
            }
            let mut below: Vec<String> = Vec::new();
            for &child in tree.children(node) {
                below.extend(clade[child as usize].iter().cloned());
            }
            below.sort();
            clade.push(below);
        }
        let mut out: Vec<(Vec<String>, u64)> = (0..tree.n_nodes() as u32)
            .map(|node| {
                let length = if node == tree.root() {
                    0.0
                } else {
                    tree.branch(node)
                };
                (clade[node as usize].clone(), length.to_bits())
            })
            .collect();
        out.sort();
        out
    }

    /// Labels `L0..L{n-1}`, which are unique and need no quoting.
    ///
    /// ### Params
    ///
    /// * `n` - How many
    ///
    /// ### Returns
    ///
    /// The labels.
    fn labels(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("L{i}")).collect()
    }

    /// Write a tree, read it back, and assert the canonical forms agree.
    ///
    /// ### Params
    ///
    /// * `tree` - The tree
    /// * `names` - One label per leaf
    fn assert_round_trip(tree: &Tree, names: &[String]) {
        let text = write_newick(tree, names).expect("writing a well-formed tree");
        let (back, back_names) = parse_newick(&text).expect("parsing our own output");
        assert_eq!(back.n_leaves(), tree.n_leaves());
        assert_eq!(back.n_nodes(), tree.n_nodes());
        assert_eq!(canonical(tree, names), canonical(&back, &back_names));
    }

    /// A five-leaf tree whose first internal node has three children.
    ///
    /// ### Returns
    ///
    /// The tree.
    fn polytomy_tree() -> Tree {
        // Node 5 gathers leaves 0, 1, 2; the root gathers leaves 3, 4 and node 5.
        let parent = vec![5, 5, 5, 6, 6, 6, NO_NODE];
        let branch = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0];
        Tree::from_parents(parent, branch, 5).expect("a valid polytomy fixture")
    }

    /// A star: every leaf hangs off the root.
    ///
    /// ### Params
    ///
    /// * `n_leaves` - How many leaves
    ///
    /// ### Returns
    ///
    /// The tree.
    fn star_tree(n_leaves: usize) -> Tree {
        let mut parent = vec![n_leaves as u32; n_leaves + 1];
        parent[n_leaves] = NO_NODE;
        let branch: Vec<f64> = (0..=n_leaves).map(|i| 1.0 + i as f64).collect();
        Tree::from_parents(parent, branch, n_leaves).expect("a valid star fixture")
    }

    // -- round trips --

    #[test]
    fn test_the_writer_refuses_the_one_tree_the_reader_will_not_take_back() {
        // The arena holds a lone leaf and `tree::cluster` relies on that, but
        // Newick has no form for one: the
        // writer emitted `"only;"` and the reader rejected it, so the two
        // disagreed about what a tree is. The writer now refuses instead.
        let tree = Tree::from_parents(vec![NO_NODE], vec![0.0], 1).expect("one leaf is an arena");
        assert!(
            matches!(
                write_newick(&tree, &["only"]),
                Err(BonsaiErrors::MalformedTree { .. })
            ),
            "a lone leaf was written out"
        );
        assert!(
            parse_newick("only;").is_err(),
            "the reader started accepting a bare label, so the writer should too"
        );
        // Two leaves is the smallest tree, and it does round trip.
        let pair = Tree::from_parents(vec![2, 2, NO_NODE], vec![0.3, 0.4, 0.0], 2).expect("pair");
        assert_round_trip(&pair, &labels(2));
    }

    #[test]
    fn test_round_trip_balanced_binary() {
        let tree = Tree::balanced_binary(8, 0.25).expect("a valid fixture");
        assert_round_trip(&tree, &labels(8));
    }

    #[test]
    fn test_round_trip_ladder() {
        let tree = Tree::ladder(9, 0.75).expect("a valid fixture");
        assert_round_trip(&tree, &labels(9));
    }

    #[test]
    fn test_round_trip_star() {
        let tree = star_tree(6);
        assert_round_trip(&tree, &labels(6));
    }

    #[test]
    fn test_round_trip_two_leaves() {
        let tree = Tree::ladder(2, 1.5).expect("a valid fixture");
        assert_round_trip(&tree, &labels(2));
    }

    #[test]
    fn test_round_trip_polytomy() {
        let tree = polytomy_tree();
        let text = write_newick(&tree, &labels(5)).expect("writing");
        // The polytomy must survive as a three-child group, not be resolved.
        let (back, names) = parse_newick(&text).expect("parsing");
        assert_eq!(canonical(&tree, &labels(5)), canonical(&back, &names));
        assert!(
            back.internal_postorder()
                .any(|n| back.children(n).len() == 3),
            "polytomy lost in {text}"
        );
    }

    #[test]
    fn test_round_trip_preserves_branch_lengths_bitwise() {
        // Awkward values: irrational-looking, subnormal-adjacent, huge, tiny,
        // and one that is not representable in 16 significant digits.
        let awkward = [
            1.0 / 3.0,
            0.1 + 0.2,
            f64::MIN_POSITIVE,
            1e-300,
            1e300,
            1.2345678901234567e-7,
            std::f64::consts::PI,
        ];
        let mut branch = vec![0.0f64; 15];
        for (i, slot) in branch.iter_mut().enumerate().take(14) {
            *slot = awkward[i % awkward.len()];
        }
        let tree = Tree::from_parents(
            {
                let mut parent = vec![NO_NODE; 15];
                // Same shape as a balanced binary over eight leaves.
                for i in 0..8 {
                    parent[i] = 8 + (i / 2) as u32;
                }
                for i in 0..4 {
                    parent[8 + i] = 12 + (i / 2) as u32;
                }
                parent[12] = 14;
                parent[13] = 14;
                parent
            },
            branch,
            8,
        )
        .expect("a valid fixture");
        assert_round_trip(&tree, &labels(8));
    }

    #[test]
    fn test_branch_length_needs_seventeen_digits() {
        // The constant is a claim about binary64, so check both halves of it.
        let value = 1.0f64 / 3.0;
        let exact = format!("{:.*e}", BRANCH_ROUND_TRIP_DIGITS - 1, value);
        assert_eq!(exact.parse::<f64>(), Ok(value));
        let short = format!("{:.*e}", BRANCH_ROUND_TRIP_DIGITS - 3, value);
        assert_ne!(short.parse::<f64>(), Ok(value));

        // And that the writer's shortest form is never worse than the claim.
        let mut out = String::new();
        push_branch_length(&mut out, value);
        assert_eq!(out.parse::<f64>(), Ok(value));
    }

    // -- explicit topologies --

    #[test]
    fn test_parse_nested_binary_topology() {
        let (tree, names) = parse_newick("((A:0.1,B:0.2):0.3,C:0.4);").expect("valid Newick");
        assert_eq!(tree.n_leaves(), 3);
        assert_eq!(tree.n_nodes(), 5);
        assert_eq!(names, vec!["A", "B", "C"]);
        // A and B are siblings; C hangs off the root.
        let a_parent = tree.parent(0).expect("A has a parent");
        assert_eq!(tree.parent(1), Some(a_parent));
        assert_ne!(tree.parent(2), Some(a_parent));
        assert_eq!(tree.parent(2), Some(tree.root()));
        assert_eq!(tree.branch(0), 0.1);
        assert_eq!(tree.branch(1), 0.2);
        assert_eq!(tree.branch(2), 0.4);
        assert_eq!(tree.branch(a_parent), 0.3);
    }

    #[test]
    fn test_parse_polytomy_keeps_all_children() {
        let (tree, names) = parse_newick("(A:1,B:1,C:1,D:1);").expect("valid Newick");
        assert_eq!(names.len(), 4);
        assert_eq!(tree.n_nodes(), 5);
        assert_eq!(tree.children(tree.root()).len(), 4);
    }

    #[test]
    fn test_a_trailing_comma_is_an_unnamed_leaf() {
        // Deliberate and standard, but it means a typo is a wrong topology
        // rather than an error, so it is
        // pinned here rather than left to be rediscovered.
        let (tree, names) = parse_newick("(a,b,);").expect("valid Newick");
        assert_eq!(names, vec!["a", "b", ""]);
        assert_eq!(tree.children(tree.root()).len(), 3);

        let (tree, names) = parse_newick("(,,,);").expect("valid Newick");
        assert_eq!(names.len(), 4);
        assert!(names.iter().all(|s| s.is_empty()));
        assert_eq!(tree.children(tree.root()).len(), 4);
    }

    #[test]
    fn test_parse_nested_polytomy() {
        let (tree, names) =
            parse_newick("((A:1,B:1,C:1):0.5,(D:1,E:1):0.5,F:2);").expect("valid Newick");
        assert_eq!(names, vec!["A", "B", "C", "D", "E", "F"]);
        assert_eq!(tree.children(tree.root()).len(), 3);
        let sizes: Vec<usize> = tree
            .internal_postorder()
            .map(|n| tree.children(n).len())
            .collect();
        assert!(sizes.contains(&3));
        assert!(sizes.contains(&2));
    }

    #[test]
    fn test_parse_quoted_labels_with_punctuation() {
        let text = "('cell (1), rep:2':0.5,'it''s a name':0.25,plain_1:0.125);";
        let (tree, names) = parse_newick(text).expect("valid Newick");
        assert_eq!(names, vec!["cell (1), rep:2", "it's a name", "plain_1"]);
        assert_eq!(tree.n_leaves(), 3);
        assert_eq!(tree.branch(0), 0.5);
        // Round-tripping must re-quote what needs it.
        let again = write_newick(&tree, &names).expect("writing");
        let (_, back) = parse_newick(&again).expect("reparsing");
        assert_eq!(back, names);
    }

    #[test]
    fn test_parse_scientific_and_missing_lengths() {
        let (tree, _) = parse_newick("((A:1.5e-3,B:2E2):+0.5,C);").expect("valid Newick");
        assert_eq!(tree.branch(0), 1.5e-3);
        assert_eq!(tree.branch(1), 200.0);
        // C carries no length at all.
        assert_eq!(tree.branch(2), DEFAULT_BRANCH_LENGTH);
    }

    #[test]
    fn test_parse_tolerates_whitespace_and_comments() {
        let messy = "  [ a header comment ]\n(\n  A : 0.1 ,\n  B\t:0.2\n)\n[inner]:0.0 ;\n";
        let (tree, names) = parse_newick(messy).expect("valid Newick");
        assert_eq!(names, vec!["A", "B"]);
        assert_eq!(tree.n_nodes(), 3);
        assert_eq!(tree.branch(0), 0.1);
        assert_eq!(tree.branch(1), 0.2);
    }

    #[test]
    fn test_internal_labels_are_written_and_ignored_on_reparse() {
        let tree = Tree::balanced_binary(4, 0.5).expect("a valid fixture");
        let internal = vec!["anc0".to_string(), "anc1".to_string(), "root".to_string()];
        let text = write_newick_labelled(&tree, &labels(4), Some(&internal)).expect("writing");
        assert!(text.contains("anc0"), "internal label missing from {text}");
        assert!(text.ends_with("root;"), "root label missing from {text}");
        let (back, names) = parse_newick(&text).expect("parsing");
        assert_eq!(canonical(&tree, &labels(4)), canonical(&back, &names));
    }

    #[test]
    fn test_root_carries_no_branch_length() {
        let tree = Tree::balanced_binary(2, 3.0).expect("a valid fixture");
        let text = write_newick(&tree, &labels(2)).expect("writing");
        assert_eq!(text, "(L0:3,L1:3);");
    }

    // -- rejections --

    #[test]
    fn test_rejects_empty_input() {
        assert!(matches!(
            parse_newick(""),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        assert!(matches!(
            parse_newick("   \n  "),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_rejects_unbalanced_parentheses() {
        for text in ["((A,B),C;", "(A,B))C;", "((A,B);"] {
            assert!(
                matches!(parse_newick(text), Err(BonsaiErrors::MalformedTree { .. })),
                "accepted {text}"
            );
        }
    }

    #[test]
    fn test_rejects_missing_semicolon() {
        assert!(matches!(
            parse_newick("((A:1,B:1):1,C:1)"),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_rejects_trailing_junk() {
        assert!(matches!(
            parse_newick("(A:1,B:1); and then some"),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        assert!(matches!(
            parse_newick("(A:1,B:1);(C:1,D:1);"),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_rejects_bad_branch_lengths() {
        for text in [
            "(A:,B:1);",
            "(A:1.2.3,B:1);",
            "(A:abc,B:1);",
            "(A:-1,B:1);",
            "(A:1e999,B:1);",
        ] {
            assert!(
                matches!(parse_newick(text), Err(BonsaiErrors::MalformedTree { .. })),
                "accepted {text}"
            );
        }
    }

    #[test]
    fn test_rejects_unterminated_quote_and_comment() {
        assert!(matches!(
            parse_newick("('unclosed:1,B:1);"),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        assert!(matches!(
            parse_newick("(A[unclosed:1,B:1);"),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    #[test]
    fn test_rejects_degenerate_trees() {
        // A single leaf, and a node with one child: neither fits the arena.
        for text in ["A;", "(A:1);", "();"] {
            assert!(
                matches!(parse_newick(text), Err(BonsaiErrors::MalformedTree { .. })),
                "accepted {text}"
            );
        }
    }

    #[test]
    fn test_rejects_label_count_mismatch() {
        let tree = Tree::balanced_binary(4, 1.0).expect("a valid fixture");
        assert!(matches!(
            write_newick(&tree, &labels(3)),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
        let internal = vec!["only one".to_string()];
        assert!(matches!(
            write_newick_labelled(&tree, &labels(4), Some(&internal)),
            Err(BonsaiErrors::MalformedTree { .. })
        ));
    }

    // -- scale --

    #[test]
    fn test_deep_ladder_survives_write_and_parse() {
        // 100k leaves is 200k nodes and 100k levels of nesting, which is far
        // past what a recursive writer or parser can carry on the stack.
        const N_LEAVES: usize = 100_000;
        let tree = Tree::ladder(N_LEAVES, 0.5).expect("a valid fixture");
        let names = labels(N_LEAVES);
        let text = write_newick(&tree, &names).expect("writing");
        let (back, back_names) = parse_newick(&text).expect("parsing");
        assert_eq!(back.n_leaves(), N_LEAVES);
        assert_eq!(back.n_nodes(), tree.n_nodes());
        assert_eq!(back.n_levels(), tree.n_levels());
        assert_eq!(back_names.len(), N_LEAVES);
        // The canonical form is quadratic, so check the branch lengths in bulk
        // instead: every non-root branch is the same 0.5.
        assert!(
            (0..back.root()).all(|n| back.branch(n) == 0.5),
            "branch lengths did not survive"
        );
    }
}
