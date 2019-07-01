
#include<fftw3.h>
#include<cstdint>
#include<memory>
#ifndef fftw_alloc_complex
#define NO_FFTW_ALLOC
#endif

namespace MeanSquareDisplacement{
  
  namespace detail{
      
    template<class real>
    struct fftw_prec_types;
    template<> struct fftw_prec_types<double>{using type = fftw_complex;};
    template<> struct fftw_prec_types<float>{using type = fftwf_complex;};

    template<class real>
    struct fftw_plan_prec;
    template<> struct fftw_plan_prec<double>{using type = fftw_plan;};
    template<> struct fftw_plan_prec<float>{using type = fftwf_plan;};
           
    template<class real> typename fftw_prec_types<real>::type* fftw_alloc_complex_prec(int N);
    template<> typename fftw_prec_types<double>::type* fftw_alloc_complex_prec<double>(int N){
#ifdef NO_FFTW_ALLOC
      return (fftw_prec_types<double>::type*) malloc(N*sizeof(fftw_prec_types<double>::type));
#else	
      return fftw_alloc_complex(N);
#endif
    }
    template<> typename fftw_prec_types<float>::type* fftw_alloc_complex_prec<float>(int N){
#ifdef NO_FFTW_ALLOC
      return (fftw_prec_types<float>::type*) malloc(N*sizeof(fftw_prec_types<double>::type));
#else	
      return fftwf_alloc_complex(N);
#endif
    }

  }


  template<class real> using fftw_complex_t = typename detail::fftw_prec_types<real>::type;
  template<class real> using fftw_plan_t = typename detail::fftw_plan_prec<real>::type;

  template <class T>
  struct FFTWallocator {
    typedef T value_type;
    FFTWallocator() = default;
    template <class U> constexpr FFTWallocator(const FFTWallocator<U>&) noexcept {}
    [[nodiscard]] T* allocate(std::size_t n) {
      if(n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
      if(auto p = (T*)(MeanSquareDisplacement::detail::fftw_alloc_complex_prec<T>((n+1)*sizeof(fftw_complex_t<T>)/sizeof(T)))) return p;
      throw std::bad_alloc();
    }
    void deallocate(T* p, std::size_t) noexcept { std::free(p); }
  };
  template <class T, class U>
  bool operator==(const FFTWallocator<T>&, const FFTWallocator<U>&) { return true; }
  template <class T, class U>
  bool operator!=(const FFTWallocator<T>&, const FFTWallocator<U>&) { return false; }



  template<class real>struct fftw_plan_many_dft_r2c_prec;
  template<>struct fftw_plan_many_dft_r2c_prec<double>{
    template <class ...T> fftw_plan_t<double> operator()(T...args){return fftw_plan_many_dft_r2c(args...);}};
  template<>struct fftw_plan_many_dft_r2c_prec<float>{
    template <class ...T> fftw_plan_t<float> operator()(T...args){return fftwf_plan_many_dft_r2c(args...);}};

  template<class real>struct fftw_plan_many_dft_c2r_prec;
  template<>struct fftw_plan_many_dft_c2r_prec<double>{
    template <class ...T> fftw_plan_t<double> operator()(T...args){return fftw_plan_many_dft_c2r(args...);}};
  template<>struct fftw_plan_many_dft_c2r_prec<float>{
    template <class ...T> fftw_plan_t<float> operator()(T...args){return fftwf_plan_many_dft_c2r(args...);}};

  void fftw_execute( fftw_plan_t<double> plan){::fftw_execute(plan);}
  void fftw_execute( fftw_plan_t<float> plan){::fftwf_execute(plan);}

  template<class ...T>void fftw_execute_dft_r2c(fftw_plan plan, T...args){::fftw_execute_dft_r2c(plan, args...);}
  template<class ...T>void fftw_execute_dft_r2c(fftwf_plan plan, T...args){fftwf_execute_dft_r2c(plan, args...);}

  template<class ...T>void fftw_execute_dft_c2r(fftw_plan plan, T...args){::fftw_execute_dft_c2r(args...);}
  template<class ...T>void fftw_execute_dft_c2r(fftwf_plan plan, T...args){fftwf_execute_dft_c2r(args...);}


}
