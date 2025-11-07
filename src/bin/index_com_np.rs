use hnsw_rs::hnsw::Hnsw;
use anndists::dist::DistL2;

use ndarray::{Array2, Axis};
use ndarray_npy::read_npy;

fn main() {
    // ============================
    // 1) Ler base de dados (N_base × 768)
    // ============================
    let base_path = "768d_uniform_data_0.npy";
    let base_array: Array2<f32> =
        read_npy(base_path).expect("Falha ao ler 768d_uniform_data_0.npy");

    let nb_elem = base_array.len_of(Axis(0));
    let dim = base_array.len_of(Axis(1));

    println!("Base lida de {base_path}: nb_elem = {nb_elem}, dim = {dim}");
    assert_eq!(dim, 768, "Dimensão da base não é 768!");

    // Converter cada linha em Vec<f32>
    let mut base_vectors: Vec<Vec<f32>> = Vec::with_capacity(nb_elem);
    for row in base_array.axis_iter(Axis(0)) {
        base_vectors.push(row.to_vec());
    }

    // ============================
    // 2) Criar índice HNSW
    // ============================
    let max_nb_connection = 16; // M
    let max_layer = 16;
    let ef_c = 200; // ef_construction

    let hnsw: Hnsw<f32, DistL2> =
        Hnsw::new(max_nb_connection, nb_elem, max_layer, ef_c, DistL2 {});

    // A versão que você está usando define:
    // pub fn parallel_insert(&self, datas: &[(&Vec<T>, usize)])
    let data_for_par_insertion: Vec<(&Vec<f32>, usize)> =
        base_vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (v, i)) // v: &Vec<f32>
            .collect();

    println!("Iniciando indexação HNSW...");
    hnsw.parallel_insert(&data_for_par_insertion);
    println!("Indexação finalizada!");

    // ============================
    // 3) Ler embeddings de consulta (N_query × 768)
    // ============================
    let query_path = "embedding_nome_corrupto.npy";
    let query_array: Array2<f32> =
        read_npy(query_path).expect("Falha ao ler embedding_nome_corrupto.npy");

    let n_query = query_array.len_of(Axis(0));
    let dim_q = query_array.len_of(Axis(1));

    println!(
        "Consultas lidas de {query_path}: n_query = {n_query}, dim = {dim_q}"
    );
    assert_eq!(
        dim_q, dim,
        "Dimensão dos embeddings de consulta é diferente da base!"
    );

    // Converter consultas em Vec<Vec<f32>>
    let mut query_vectors: Vec<Vec<f32>> = Vec::with_capacity(n_query);
    for row in query_array.axis_iter(Axis(0)) {
        query_vectors.push(row.to_vec());
    }

    // ============================
    // 4) Buscar k vizinhos mais próximos para cada consulta
    // ============================
    let knbn = 10;                // top-k vizinhos
    let ef_search = max_nb_connection; // pode aumentar, ex.: 100, 200

    let max_print = n_query; // para não imprimir tudo

    for (qi, qv) in query_vectors.iter().take(max_print).enumerate() {
        let vizinhos = hnsw.search(qv, knbn, ef_search);

        println!("\nQuery #{qi} — top {knbn} vizinhos:");
        for n in vizinhos {
            println!("  id = {}, dist = {}", n.get_origin_id(), n.get_distance());
        }
    }

    println!("\nTotal de consultas processadas (em memória): {n_query}");
}
