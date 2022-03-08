use image::{Pixel, Rgb};
use rand::Rng;

const dft_size:usize = 16;

fn main()  {

    let img = image::open("sample/test.jpg").unwrap();
    let raw_img = img.into_rgba8();
    //let raw_img = dft(raw_img);
    let raw_img = pixel_k_means(raw_img, 5);
    raw_img.save("output/test.png").unwrap();
}

fn color_compress(mut img : image::RgbaImage, threshold : u8) -> image::RgbaImage {

 
    let pixel_iter = img.pixels_mut();
    // Iterate through pixels and compress the color space
    for pixel in pixel_iter {
        pixel.apply(|t| ->u8 {
            t / threshold * threshold
        })
    }
    
    return img
    
}

fn dft(mut img : image::RgbaImage) -> image::RgbaImage {

    let (width, height) = img.dimensions();
    let col = width / dft_size as u32;
    let row = height / dft_size as u32;
    // Initialize DFT array
    for i in 0..col {
        for j in 0..row {
            dft_block(&mut img,i,j);
        }
    }
    return img
}

fn dft_block( img : &mut image::RgbaImage, i :u32, j:u32) {
    
    let (img_width, img_height) = img.dimensions();
    let mut w_end = img_width;
    let mut h_end = img_height;
    if w_end > (i + 1 as u32) * dft_size as u32 {
        w_end = (i + 1) * dft_size as u32;
    }
    if h_end > (j + 1 as u32) * dft_size as u32 {
        h_end = (j + 1) * dft_size as u32;
    }
    let w_begin = i * dft_size as u32;
    let h_begin = j * dft_size as u32;
    let width = w_end - w_begin;
    let height = h_end - h_begin;
    // Initialize DFT array
    let mut r_r :[f64; dft_size*dft_size] = [0.0; dft_size*dft_size];
    let mut r_i :[f64; dft_size*dft_size] = [0.0; dft_size*dft_size];
    let mut g_r :[f64; dft_size*dft_size] = [0.0; dft_size*dft_size];
    let mut g_i :[f64; dft_size*dft_size] = [0.0; dft_size*dft_size];
    let mut b_r :[f64; dft_size*dft_size] = [0.0; dft_size*dft_size];
    let mut b_i :[f64; dft_size*dft_size] = [0.0; dft_size*dft_size];

    // Phase 1 - Calculate the DFT table of image
    for x in w_begin..w_end {
        for y in h_begin..h_end {

            for u in 0..dft_size {
                for v in 0..dft_size {
                    let color = img.get_pixel(x,y);
                    let angle = 2.0 * std::f64::consts::PI * (((u ) as f64) * ((x + 1) as f64) / (width as f64) + ( (v ) as f64) * ((y + 1) as f64) / (height as f64));
                    r_r[u * dft_size + v] += angle.cos() * color[0] as f64;
                    r_i[u * dft_size + v] += - angle.sin() * color[0] as f64;
                    g_r[u * dft_size + v] += angle.cos() * color[1] as f64;
                    g_i[u * dft_size + v] += - angle.sin() * color[1] as f64;
                    b_r[u * dft_size + v] += angle.cos() * color[2] as f64;
                    b_i[u * dft_size + v] += - angle.sin() * color[2] as f64;
                    
                }
            }
        }
    }
    // Phase 2 - Clear the high freq
    for u in 2..dft_size {
        for v in 2.. dft_size {
            r_r[u * dft_size + v] = 0.0;
            r_i[u * dft_size + v] = 0.0;
            g_r[u * dft_size + v] = 0.0;
            g_i[u * dft_size + v] = 0.0;
            b_r[u * dft_size + v] = 0.0;
            b_i[u * dft_size + v] = 0.0;
        }
    }
    // Phase 3 - Regenerate the image from DFT table
    for x in w_begin..w_end {
        for y in h_begin..h_end {
            let mut r = 0.0_f64;
            let mut g = 0.0_f64;
            let mut b = 0.0_f64;
            
            for u in 0..dft_size {
                for v in 0..dft_size {
                    let angle = 2.0 * std::f64::consts::PI * (((u ) as f64) * ((x + 1) as f64) / (width as f64) + ( (v ) as f64) * ((y + 1) as f64) / (height as f64));
                    r += angle.cos() * r_r[u*dft_size + v] - angle.sin()*r_i[u*dft_size + v];
                    g += angle.cos() * g_r[u*dft_size + v] - angle.sin()*g_i[u*dft_size + v];
                    b += angle.cos() * b_r[u*dft_size + v] - angle.sin()*b_i[u*dft_size + v];

                }
            }
            r = r / width as f64 / height as f64;
            g = g / width as f64 / height as f64;
            b = b / width as f64 / height as f64;
            let old_color = img.get_pixel_mut(x,y).channels_mut();
            old_color[0] = r as u8;
            old_color[1] = g as u8;
            old_color[2] = b as u8;
        }
    }
}

fn pixel_k_means(mut img: image::RgbaImage, k:u8) -> image::RgbaImage {
    
    let mut rng = rand::thread_rng();
    let (width, height) = img.dimensions();
    // Initialize Label and mean
    let mut label: Vec<u8> = vec![0;(width * height) as usize];//Vec::with_capacity(width as usize * height as usize);
    let mut kernels: Vec<image::Rgb<u8>> = Vec::with_capacity(k as usize);
    let mut moment: Vec<image::Rgb<u32>> = Vec::with_capacity(k as usize);
    let mut count: Vec<u32> = Vec::with_capacity(k as usize);
    for _i in 0..k {
        let r: u8 = rng.gen();
        let g: u8 = rng.gen();
        let b: u8 = rng.gen();
        kernels.insert(0, image::Rgb::from([r,g,b]));
        moment.insert(0, image::Rgb::from([0,0,0]));
        count.insert(0,0);

    }
    for _i in 0..10 {
        for j in 0..k {
            moment[j as usize][0] = 0;
            moment[j as usize][1] = 0;
            moment[j as usize][2] = 0;
            count[j as usize] = 0;

        }
        for x in 0..width {
            for y in 0..height {
                let old_color = img.get_pixel(x, y).channels();
                let mut min_distance: i32 = i32::MAX;
                let mut min_index = k;
                for i in 0..k {
                    let distance = (old_color[0] as i32 - kernels[i as usize][0] as i32).pow(2) + (old_color[1] as i32 - kernels[i as usize][1] as i32).pow(2) + (old_color[2] as i32 - kernels[i as usize][2] as i32).pow(2);
                   if distance < min_distance {
                        min_distance = distance;
                        min_index = i;
                   }
                }
                if min_index == k {
                    panic!();
                }
                count[min_index as usize] += 1;
                moment[min_index as usize][0] += old_color[0] as u32;
                moment[min_index as usize][1] += old_color[1] as u32;
                moment[min_index as usize][2] += old_color[2] as u32;
                label[(x * height + y) as usize] = min_index;
            }
        }
        for j in 0..k {
            if count[j as usize] == 0 {
                kernels[j as usize][0] = 0;
                kernels[j as usize][1] = 0;
                kernels[j as usize][2] = 0;
            }
            else {
                kernels[j as usize][0] = (moment[j as usize][0] / count[j as usize]) as u8;
                kernels[j as usize][1] = (moment[j as usize][1] / count[j as usize]) as u8;
                kernels[j as usize][2] = (moment[j as usize][2] / count[j as usize]) as u8;
            }
        }
    }
    for x in 0..width {
        for y in 0..height {
            let new_color = &kernels[label[(x * height + y) as usize] as usize];
            img.put_pixel(x, y, image::Rgba::from([new_color[0],new_color[1],new_color[2],255]));
        }
    }
    for i in 0..k {
        println!("{:?}", kernels[i as usize]);
    }
    return img
}
