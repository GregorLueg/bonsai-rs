//! Writing trees and layouts out as CSV, for plotting elsewhere.
//!
//! This crate computes coordinates and stops there. Rendering belongs in
//! whatever the caller already plots with, so the contract is a table rather
//! than an image format.
//!
//! One row per node carries everything a plot needs: its position, its parent,
//! and the branch length between them. Edges are the rows with a parent, so a
//! self-join on `parent` gives the segments and no second file is needed.

use std::fmt::Write as _;

use crate::errors::BonsaiErrors;
use crate::tree::Tree;
use crate::tree::layout::Layout;

/// Quote a label for CSV if it needs it.
///
/// Cell barcodes are usually bare alphanumerics, but nothing stops a caller
/// passing something with a comma or a quote in it, and silently writing a
/// broken table is worse than the small cost of checking.
///
/// ### Params
///
/// * `label` - The label
///
/// ### Returns
///
/// The label, quoted and escaped only if it has to be.
fn escape(label: &str) -> String {
    if label.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", label.replace('"', "\"\""))
    } else {
        label.to_string()
    }
}

/// Write a tree and a layout as one CSV table.
///
/// Columns: `node`, `parent`, `is_leaf`, `label`, `branch`, `x`, `y`. The root's
/// `parent` is empty and its `branch` is zero. Internal nodes have an empty
/// `label` unless one was supplied.
///
/// Plotting the edges is a self-join: every row with a `parent` is a segment
/// from its own `(x, y)` to its parent's.
///
/// ### Params
///
/// * `tree` - The tree
/// * `layout` - Coordinates from [`crate::tree::layout`], one per node
/// * `leaf_labels` - Label per leaf, indexed by leaf id; empty for none
///
/// ### Returns
///
/// The CSV text, or `NodeOutOfRange` if the layout does not cover the tree, or
/// `ShapeMismatch` if the labels do not match the leaf count.
pub fn layout_csv<S: AsRef<str>>(
    tree: &Tree,
    layout: &Layout,
    leaf_labels: &[S],
) -> Result<String, BonsaiErrors> {
    let n = tree.n_nodes();
    if layout.n_nodes() != n {
        return Err(BonsaiErrors::NodeOutOfRange {
            index: layout.n_nodes(),
            n_nodes: n,
        });
    }
    if !leaf_labels.is_empty() && leaf_labels.len() != tree.n_leaves() {
        return Err(BonsaiErrors::ShapeMismatch {
            mean_cells: tree.n_leaves(),
            mean_features: 1,
            sd_cells: leaf_labels.len(),
            sd_features: 1,
        });
    }

    let mut out = String::with_capacity(n * 48);
    out.push_str("node,parent,is_leaf,label,branch,x,y\n");
    for node in 0..n as u32 {
        let is_leaf = (node as usize) < tree.n_leaves();
        let label = if is_leaf && !leaf_labels.is_empty() {
            escape(leaf_labels[node as usize].as_ref())
        } else {
            String::new()
        };
        match tree.parent(node) {
            Some(up) => {
                let _ = writeln!(
                    out,
                    "{node},{up},{is_leaf},{label},{},{},{}",
                    tree.branch(node),
                    layout.x[node as usize],
                    layout.y[node as usize]
                );
            }
            None => {
                let _ = writeln!(
                    out,
                    "{node},,{is_leaf},{label},0,{},{}",
                    layout.x[node as usize], layout.y[node as usize]
                );
            }
        }
    }
    Ok(out)
}

/// Write per-node posterior positions as one CSV table.
///
/// Columns: `node`, `is_leaf`, then `mean_<f>` and `sd_<f>` for each retained
/// feature, using the caller's original feature indices so a column can be
/// matched back to a gene.
///
/// Wide rather than long, because a caller wanting a feature reads one column,
/// and a long table of `n_nodes * n_features` rows is unwieldy past a few
/// hundred features. Values are in raw units, as
/// [`crate::bonsai::BonsaiResult`] returns them.
///
/// ### Params
///
/// * `tree` - The tree the positions belong to
/// * `means` - Posterior means, row-major `[node][feature]`
/// * `sds` - Posterior standard deviations, same layout
/// * `features` - Original feature index per column
///
/// ### Returns
///
/// The CSV text, or `ShapeMismatch` if the blocks do not agree with the tree and
/// the feature list.
pub fn posteriors_csv<T: std::fmt::Display>(
    tree: &Tree,
    means: &[T],
    sds: &[T],
    features: &[usize],
) -> Result<String, BonsaiErrors> {
    let (n, p) = (tree.n_nodes(), features.len());
    if means.len() != n * p || sds.len() != n * p {
        return Err(BonsaiErrors::ShapeMismatch {
            mean_cells: means.len() / p.max(1),
            mean_features: p,
            sd_cells: n,
            sd_features: p,
        });
    }

    let mut out = String::with_capacity(n * p * 16);
    out.push_str("node,is_leaf");
    for &f in features {
        let _ = write!(out, ",mean_{f},sd_{f}");
    }
    out.push('\n');

    for node in 0..n {
        let _ = write!(out, "{node},{}", node < tree.n_leaves());
        let lo = node * p;
        for g in 0..p {
            let _ = write!(out, ",{},{}", means[lo + g], sds[lo + g]);
        }
        out.push('\n');
    }
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::NO_NODE;
    use crate::tree::layout::equal_angle;

    fn fixture() -> (Tree, Layout) {
        let tree = Tree::from_parents(
            vec![4, 4, 5, 5, 6, 6, NO_NODE],
            vec![1.0, 2.0, 3.0, 4.0, 0.5, 0.5, 0.0],
            4,
        )
        .expect("fixture");
        let layout = equal_angle(&tree, None).expect("layout");
        (tree, layout)
    }

    #[test]
    fn test_layout_csv_has_a_row_per_node_and_a_header() {
        let (tree, layout) = fixture();
        let csv = layout_csv(&tree, &layout, &["a", "b", "c", "d"]).expect("csv");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), tree.n_nodes() + 1);
        assert_eq!(lines[0], "node,parent,is_leaf,label,branch,x,y");
        // Every data row has the same column count as the header.
        let want = lines[0].matches(',').count();
        for line in &lines[1..] {
            assert_eq!(line.matches(',').count(), want, "ragged row: {line}");
        }
    }

    #[test]
    fn test_the_root_has_an_empty_parent_and_leaves_carry_their_labels() {
        let (tree, layout) = fixture();
        let csv = layout_csv(&tree, &layout, &["a", "b", "c", "d"]).expect("csv");
        let lines: Vec<&str> = csv.lines().collect();

        let root = lines[tree.root() as usize + 1];
        let cells: Vec<&str> = root.split(',').collect();
        assert_eq!(cells[1], "", "the root should have no parent");
        assert_eq!(cells[2], "false");

        assert!(
            lines[1].starts_with("0,4,true,a,"),
            "leaf row was {}",
            lines[1]
        );
    }

    #[test]
    fn test_labels_needing_quotes_get_them() {
        let (tree, layout) = fixture();
        let csv =
            layout_csv(&tree, &layout, &["plain", "has,comma", "has\"quote", "d"]).expect("csv");
        assert!(csv.contains("\"has,comma\""));
        assert!(csv.contains("\"has\"\"quote\""));
        // A label needing nothing is left alone.
        assert!(csv.contains(",plain,"));
    }

    #[test]
    fn test_labels_are_optional() {
        let (tree, layout) = fixture();
        let csv = layout_csv::<&str>(&tree, &layout, &[]).expect("csv");
        assert_eq!(csv.lines().count(), tree.n_nodes() + 1);
    }

    #[test]
    fn test_mismatched_labels_and_layouts_are_rejected() {
        let (tree, layout) = fixture();
        assert!(layout_csv(&tree, &layout, &["only", "three", "here"]).is_err());

        let short = Layout {
            x: vec![0.0; 3],
            y: vec![0.0; 3],
        };
        assert!(layout_csv::<&str>(&tree, &short, &[]).is_err());
    }

    #[test]
    fn test_posteriors_csv_names_columns_by_original_feature() {
        let (tree, _) = fixture();
        let n = tree.n_nodes();
        let features = vec![3usize, 17, 42];
        let means: Vec<f64> = (0..n * 3).map(|i| i as f64).collect();
        let sds = vec![0.5f64; n * 3];

        let csv = posteriors_csv(&tree, &means, &sds, &features).expect("csv");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "node,is_leaf,mean_3,sd_3,mean_17,sd_17,mean_42,sd_42"
        );
        assert_eq!(lines.len(), n + 1);
        assert!(lines[1].starts_with("0,true,0,0.5,1,0.5,2,0.5"));
    }

    #[test]
    fn test_posteriors_csv_rejects_a_block_of_the_wrong_shape() {
        let (tree, _) = fixture();
        let features = vec![0usize, 1];
        let wrong = vec![0.0f64; 5];
        assert!(posteriors_csv(&tree, &wrong, &wrong, &features).is_err());
    }
}
