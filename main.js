const app = Vue.createApp({
    data() {
        return {
            tradingResult: [
                {date: "2020/11/09", action: 'Buy', profit: 1111},
                {date: "2020/11/09", action: 'Buy', profit: 1111},
                {date: "2020/11/09", action: 'Buy', profit: 1111},
                {date: "2020/11/09", action: 'Buy', profit: 1111},
                {date: "2020/11/09", action: 'Buy', profit: 1111},
                {date: "2020/11/09", action: 'Buy', profit: 1111},
                {date: "2020/11/09", action: 'Buy', profit: 1111}
            ],
            obtainedDQNResult: true
        }
    },
    methods: {
        receivedDQNResult(){
            this.obtainedDQNResult = true
        }
    },
    computed: {
        showDQNResult(){
            // 0 = true, otherwise false
            return this.obtainedDQNResult
        }
    }

})