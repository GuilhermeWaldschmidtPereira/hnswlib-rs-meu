use hnsw_rs::hnsw::Hnsw;
use anndists::dist::DistL2;

fn main() {
    let dim = 32;
    let nb_elem = 1000;

    let mut meus_dados: Vec<Vec<f32>> = Vec::with_capacity(nb_elem);
    for i in 0..nb_elem {
        let mut v = Vec::with_capacity(dim);
        for j in 0..dim {
            v.push((i + j) as f32);
        }
        meus_dados.push(v);
    }

    let max_nb_connection = 16;
    let max_layer = 16;
    let ef_c = 200;

    let hnsw: Hnsw<f32, DistL2> =
        Hnsw::new(max_nb_connection, nb_elem, max_layer, ef_c, DistL2 {});

    // aqui está o formato que a função espera: &[(&Vec<f32>, usize)]
    let data_for_par_insertion: Vec<(&Vec<f32>, usize)> =
        meus_dados
            .iter()
            .enumerate()
            .map(|(i, v)| (v, i))
            .collect();

    hnsw.parallel_insert(&data_for_par_insertion);

    println!("Indexação finalizada!");

    let knbn = 10;
    let ef_search = max_nb_connection;
    let query = &meus_dados[0];          // &Vec<f32>
    let vizinhos = hnsw.search(query, knbn, ef_search);

    for n in vizinhos {
        println!("id = {}, dist = {}", n.get_origin_id(), n.get_distance());
    }
}
