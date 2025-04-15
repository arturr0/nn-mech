#define _USE_MATH_DEFINES
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <thread>
#include <mutex>
#include <allegro5/allegro.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_ttf.h>
#include <allegro5/allegro_font.h>
#include <sstream>

template <typename T>
int sign(T val)
{
    return (T(0) < val) - (val < T(0));
}

using namespace std;

inline double activation(double x) { return tanh(x); }
inline double activation_derivative(double x) { return 1 - x * x; }
inline double normalize(double x) { return x / 20; }
inline double denormalize(double x) { return x * 20; }

double total_loss = 0;
int train_epoch = 0;
int max_epoch = 100;

mutex mtx; // For thread safety

// Neural network for learning f(x)=A*sin(x)+B*cos(x)
// Input: { norm_A, sin(x), norm_B, cos(x) }
// Output: vector<double> of size 2, where [0]=predicted A and [1]=predicted B.
class NeuralNetwork
{
public:
    vector<vector<double>> W1, W2;
    vector<double> b1, b2;
    int input_size, hidden_size, output_size;
    double learning_rate;

    vector<vector<double>> last_W1, last_W2;
    vector<double> last_b1, last_b2;

    NeuralNetwork(int in_size, int hid_size, int out_size, double lr = 0.001)
        : input_size(in_size), hidden_size(hid_size), output_size(out_size), learning_rate(lr)
    {
        initialize_weights();
    }

    void initialize_weights()
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-0.5, 0.5);

        W1.resize(hidden_size, vector<double>(input_size));
        W2.resize(output_size, vector<double>(hidden_size));
        b1.resize(hidden_size);
        b2.resize(output_size);

        for (auto& row : W1)
            for (auto& w : row)
                w = dist(gen);
        for (auto& row : W2)
            for (auto& w : row)
                w = dist(gen);
        for (auto& b : b1)
            b = dist(gen);
        for (auto& b : b2)
            b = dist(gen);

        last_W1 = W1;
        last_W2 = W2;
        last_b1 = b1;
        last_b2 = b2;
    }

    // Forward returns a vector: [predicted A, predicted B]
    vector<double> forward(const vector<double>& x, vector<double>& h, bool mode)
    {
        h.resize(hidden_size);
        for (int i = 0; i < hidden_size; i++)
        {
            h[i] = b1[i];
            for (int j = 0; j < input_size; j++)
                h[i] += W1[i][j] * x[j];
            h[i] = activation(h[i]);
        }
        vector<double> y(output_size);
        for (int k = 0; k < output_size; k++)
        {
            y[k] = b2[k];
            for (int i = 0; i < hidden_size; i++)
                y[k] += W2[k][i] * h[i];
        }
        if (mode)
            cout << "Predicted A: " << y[0] << ", Predicted B: " << y[1] << endl;
        return y;
    }

    void store_last_parameters()
    {
        last_W1 = W1;
        last_W2 = W2;
        last_b1 = b1;
        last_b2 = b2;
    }
    void restore_last_parameters()
    {
        W1 = last_W1;
        W2 = last_W2;
        b1 = last_b1;
        b2 = last_b2;
    }

    // In each training sample, we compute f_pred = (predicted A)*sin(x) + (predicted B)*cos(x)
    // and compare it to the target.
    void train_sample(const vector<vector<double>>& X, const vector<double>& Y, int sample_start, int sample_end)
    {
        double local_loss = 0.0;
        for (int sample = sample_start; sample < sample_end; sample++)
        {
            vector<double> h;
            vector<double> pred = forward(X[sample], h, false); // pred[0] = A, pred[1] = B
            // Extract sin(x) and cos(x) from input:
            double s = X[sample][1]; // sin(x)
            double c = X[sample][3]; // cos(x)
            double f_pred = pred[0] * s + pred[1] * c;
            double error = f_pred - Y[sample];
            local_loss += error * error;

            // Compute gradients for output layer:
            // For neuron 0 (A): derivative factor = sin(x)
            // For neuron 1 (B): derivative factor = cos(x)
            vector<double> grad_out(2);
            grad_out[0] = error * s;
            grad_out[1] = error * c;
            vector<vector<double>> dW2(output_size, vector<double>(hidden_size, 0.0));
            for (int k = 0; k < output_size; k++)
            {
                for (int i = 0; i < hidden_size; i++)
                {
                    dW2[k][i] = grad_out[k] * h[i];
                }
            }
            vector<double> db2 = grad_out;

            // Backpropagate into hidden layer:
            vector<double> dH(hidden_size, 0.0);
            for (int i = 0; i < hidden_size; i++)
            {
                double sum = 0.0;
                for (int k = 0; k < output_size; k++)
                {
                    sum += W2[k][i] * grad_out[k];
                }
                dH[i] = sum * activation_derivative(h[i]);
            }
            vector<vector<double>> dW1(hidden_size, vector<double>(input_size, 0.0));
            for (int i = 0; i < hidden_size; i++)
            {
                for (int j = 0; j < input_size; j++)
                {
                    dW1[i][j] = dH[i] * X[sample][j];
                }
            }
            vector<double> db1 = dH;

            // Update weights:
            for (int k = 0; k < output_size; k++)
            {
                for (int i = 0; i < hidden_size; i++)
                {
                    W2[k][i] -= learning_rate * dW2[k][i];
                }
                b2[k] -= learning_rate * db2[k];
            }
            for (int i = 0; i < hidden_size; i++)
            {
                for (int j = 0; j < input_size; j++)
                {
                    W1[i][j] -= learning_rate * dW1[i][j];
                }
                b1[i] -= learning_rate * db1[i];
            }
        }
        lock_guard<mutex> guard(mtx);
        total_loss += local_loss;
    }

    void train(const vector<vector<double>>& X, const vector<double>& Y, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            total_loss = 0.0;
            vector<thread> threads;
            size_t batch_size = X.size() / 4;
            for (int i = 0; i < 4; i++)
            {
                size_t sample_start = i * batch_size;
                size_t sample_end = (i == 3) ? X.size() : (i + 1) * batch_size;
                threads.push_back(thread(&NeuralNetwork::train_sample, this, ref(X), ref(Y), sample_start, sample_end));
            }
            for (auto& t : threads)
                t.join();
            if (isnan(total_loss) || total_loss > 1e10)
            {
                restore_last_parameters();
                learning_rate *= 0.9;
            }
            else
            {
                store_last_parameters();
            }
            train_epoch++;
            if (train_epoch > max_epoch)
                break;
        }
    }

    // For a set of input points, compute the average predicted A and B.
    // The network output is [A, B] and we combine with sin(x) and cos(x) to compute f(x).
    double predict_from_sin_cos_points(const vector<vector<double>>& sin_values)
    {
        double sum_A = 0.0, sum_B = 0.0;
        for (const auto& x : sin_values)
        {
            vector<double> h;
            vector<double> pred = forward(x, h, true); // pred[0]=A, pred[1]=B
            sum_A += pred[0];
            sum_B += pred[1];
        }
        double avg_A = sum_A / sin_values.size();
        double avg_B = sum_B / sin_values.size();
        cout << "Average predicted A: " << denormalize(avg_A)
            << ", Average predicted B: " << denormalize(avg_B) << endl;
        return denormalize(avg_A); // or you can return both if needed
    }
};

// Generate training data for f(x) = A*sin(x) + B*cos(x).
// For each sample, the input is { norm_A, sin(x), norm_B, cos(x) } and the target is f(x).
void generate_data(vector<vector<double>>& X, vector<double>& Y)
{
    const double frequency = 2.0; // 2 full periods over the range [0, 2π]
    for (int A = -20; A <= 20; ++A)
    {
        for (int B = -20; B <= 20; ++B)
        {
            double norm_A = normalize(A);
            double norm_B = normalize(B);
            for (double x = 0; x <= 4 * M_PI; x += 0.1)
            {
                double fx = norm_A * sin(frequency * x) + norm_B * cos(frequency * x);
                X.push_back({ norm_A, sin(frequency * x), norm_B, cos(frequency * x) });
                Y.push_back(fx);
            }
        }
    }
}

// Render training process with Allegro5 (animation, chart, amplitude info).
void render_training(NeuralNetwork& nn, const vector<vector<double>>& X, const vector<double>& Y)
{
    if (!al_init() || !al_init_primitives_addon() || !al_install_keyboard())
        return;

    // Get screen size information
    ALLEGRO_MONITOR_INFO info;
    al_get_monitor_info(0, &info);
    int screen_width = info.x2 - info.x1;
    int screen_height = info.y2 - info.y1;

    // Create display with full screen size
    ALLEGRO_DISPLAY* display = al_create_display(screen_width, screen_height);
    ALLEGRO_EVENT_QUEUE* event_queue = al_create_event_queue();
    al_init_font_addon();
    al_init_ttf_addon();
    ALLEGRO_FONT* font = al_load_ttf_font("VeraBd.ttf", 14, 0);      // Smaller font for axis
    ALLEGRO_FONT* font_axis = al_load_ttf_font("VeraBd.ttf", 11, 0); // Smaller font for axis
    if (!font)
    {
        cerr << "Failed to load font!" << endl;
        return;
    }
    al_register_event_source(event_queue, al_get_keyboard_event_source());
    if (!display)
        return;

    bool running = true;
    const int cols = 2, rows = 2; // Number of columns and rows in the grid
    const int top_margin = 3;     // Space at the top for text (like epoch info)

    // Calculate the width and height of each cell in the grid
    int cell_w = screen_width / cols;
    int cell_h = (screen_height - top_margin - 50) / rows; // Reduced height for better fit

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dist(0, 1680);
    vector<int> selected_samples(4);
    for (int i = 0; i < 4; i++)
        selected_samples[i] = dist(gen);

    while (running && train_epoch <= max_epoch)
    {
        ALLEGRO_EVENT event;
        while (al_get_next_event(event_queue, &event))
        {
            if (event.type == ALLEGRO_EVENT_KEY_DOWN)
            {
                if (event.keyboard.keycode == ALLEGRO_KEY_ESCAPE)
                    running = false;
                else if (event.keyboard.keycode == ALLEGRO_KEY_UP)
                    nn.learning_rate *= 1.1;
                else if (event.keyboard.keycode == ALLEGRO_KEY_DOWN)
                    nn.learning_rate *= 0.9;
            }
        }

        nn.train(X, Y, 1);
        al_clear_to_color(al_map_rgb(15, 15, 20));
        al_draw_textf(font, al_map_rgb(255, 255, 255), 10, 3, 0,
            "Epoch: %d | Avg Loss: %.6f | Learning Rate: %.4f", train_epoch, total_loss / X.size(), nn.learning_rate);

        // Draw grid of samples with predicted info (simplified)
        for (int idx = 0; idx < 4; idx++)
        {
            int row = idx / cols;
            int col = idx % cols;
            int x_offset = col * cell_w;
            int y_offset = row * cell_h + top_margin;

            // Draw background grid lines
            for (int i = 0; i < 16; ++i)
            {
                int gx = x_offset + (i * cell_w) / 10;
                int gy = y_offset + (i * cell_h) / 16;
                al_draw_line(gx, y_offset, gx, y_offset + cell_h, al_map_rgb(40, 40, 60), 1);
                al_draw_line(x_offset, gy, x_offset + cell_w, gy, al_map_rgb(40, 40, 60), 1);
            }

            // Draw the axes at the center of each cell
            al_draw_line(x_offset, y_offset + cell_h / 2, x_offset + cell_w, y_offset + cell_h / 2, al_map_rgb(100, 100, 100), 2);
            al_draw_line(x_offset + cell_w / 2, y_offset, x_offset + cell_w / 2, y_offset + cell_h, al_map_rgb(100, 100, 100), 2);

            // Draw ticks and labels on X-axis (centered around 0)
            for (int i = -5; i <= 5; ++i) // X-axis from -π to π
            {
                if (i == 5 && col == 0)
                    continue;
                double x_val = ((i + 5) * cell_w) / 10.0;                                                                                                              // Center 0 at the middle
                al_draw_textf(font_axis, al_map_rgb(255, 255, 255), x_offset + x_val, y_offset + cell_h / 2 + 10, ALLEGRO_ALIGN_CENTER, "%.2f", (i / 5.0) * 4 * M_PI); // Range: -π to π
            }

            // Draw ticks and labels on Y-axis (adjusted range from -28 to 28)
            for (int i = -8; i <= 8; ++i) // Y-axis from -28 to 28
            {
                double y_val = (cell_h / 2) - i * (cell_h / 16.0); // Adjusted for amplitude scaling
                if (i == -10 && row == 0)
                    continue;
                if (i != 0)
                {
                    al_draw_textf(font_axis, al_map_rgb(255, 255, 255), x_offset + 10 + cell_w / 2, y_offset + y_val, ALLEGRO_ALIGN_LEFT, "%.2f", (i / 8.0) * 28); // Range: -28 to 28
                }
            }

            // Variables to track prediction and real values for each sample
            double prev_x_pred = -1, prev_y_pred = -1;
            double prev_x_real = -1, prev_y_real = -1;
            size_t start_index = selected_samples[idx] * (Y.size() / 1681);
            size_t end_index = start_index + (Y.size() / 1681);
            double max_pred = -1;
            double A, B;

            // Process and draw predicted values
            for (size_t i = start_index; i < end_index; i++)
            {
                vector<double> h;
                vector<double> pred = nn.forward(X[i], h, false);
                double s = X[i][1], c = X[i][3];
                double f_pred = pred[0] * s + pred[1] * c;
                double x_value = (i - start_index) * 0.1; // assuming step 0.05
                A = X[i][0];
                B = X[i][2];
                double total_range = 4 * M_PI;
                double norm_x = x_value / total_range;
                double x = x_offset + norm_x * cell_w;

                double y_real = y_offset + (cell_h / 2) - Y[i] * (cell_h / 2.8);        // Adjusted scaling for amplitude ±28
                double y_predicted = y_offset + (cell_h / 2) - f_pred * (cell_h / 2.8); // Same adjustment for predicted values

                if (prev_x_pred != -1)
                    al_draw_line(prev_x_pred, prev_y_pred, x, y_predicted, al_map_rgb(0, 255, 0), 3);
                if (prev_x_real != -1)
                    al_draw_line(prev_x_real, prev_y_real, x, y_real, al_map_rgb(255, 255, 255), 2);

                prev_x_pred = x;
                prev_y_pred = y_predicted;
                prev_x_real = x;
                prev_y_real = y_real;
            }

            // Display sample information
            al_draw_textf(font, al_map_rgb(180, 180, 255), x_offset + 10, y_offset + 20, 0,
                "A = %.2f | B = %.2f | C = %.2f", A * 20, B * 20, sqrt(A * A * 400 + B * B * 400));
        }

        // Update display after drawing
        al_flip_display();
    }

    // Clean up resources
    al_destroy_display(display);
    al_destroy_event_queue(event_queue);
    al_destroy_font(font);
}

// Generate sine and cosine points for prediction given new amplitudes A and B.
vector<vector<double>> generate_sin_cos_points(double A, double B)
{
    double norm_A = normalize(A);
    double norm_B = normalize(B);
    vector<vector<double>> points;
    for (double x = 0; x <= 4 * M_PI; x += 0.1)
    {
        points.push_back({ norm_A, sin(x), norm_B, cos(x) });
    }
    return points;
}

int main()
{
    vector<vector<double>> X;
    vector<double> Y;
    generate_data(X, Y);

    // Create neural network with input dimension 4, hidden neurons 15, output dimension 2 (predicting A and B)
    NeuralNetwork nn(4, 15, 2, 0.001);

    // Start training rendering in a separate thread so that we can visualize the learning process.
    thread renderThread(render_training, ref(nn), ref(X), ref(Y));

    // Let the network train/render for a while (adjust duration as needed)
    // this_thread::sleep_for(chrono::seconds(5));

    // After some training, interactively predict using user-specified amplitudes.
    double new_A, new_B;
    cout << "Enter new amplitudes (A and B): ";
    cin >> new_A >> new_B;
    vector<vector<double>> sin_points = generate_sin_cos_points(new_A, new_B);
    double predicted_A = nn.predict_from_sin_cos_points(sin_points);
    cout << "Predicted A from sine points: " << predicted_A << endl;

    renderThread.join();
    return 0;
}
